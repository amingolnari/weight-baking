"""The edit specification: tolerant parsing, repair, strength scaling, the affine map it compiles to, the rule-based
proposal and the closed-form ridge solution that bakes an edit into one weight matrix."""

import ast
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


def heuristic_spec(L: int, d: Dict[str, pd.DataFrame], n: int = 3) -> Dict[str, Any]:
    """Rule-based proposal with ADAPTIVE factors: stretch the n most conflated category pairs, each by the factor that would
    bring its centroid distance up to the median over all pairs (cosine distance grows roughly with the square of the gap).
    """
    p = d["category_pairs"]
    med = p.centroid_cos_dist.median()
    top = p.sort_values("cross_knn_rate", ascending=False).head(n)
    f = lambda r: round(float(np.clip(np.sqrt(med / max(r.centroid_cos_dist, 1e-6)), 1.05, 2.0)), 3)
    return {
        "layer": L,
        "preserve_norm": True,
        "rationale": f"rule: {n} most conflated pairs, factor = sqrt(median distance / pair distance)",
        "ops": [{"op": "separate", "a": r.a, "b": r.b, "factor": f(r)} for r in top.itertuples()],
    }


def parse_json(raw: str) -> Any:
    """Tolerant parser: first JSON object in the reply; survives code fences, trailing commas, single quotes and a cut-off ending."""
    s = re.sub(r"```(?:json)?", "", raw)
    i = s.find("{")
    if i < 0:
        raise ValueError("no JSON object found in the reply")
    depth, instr, esc, end = 0, False, False, None
    for j in range(i, len(s)):
        ch = s[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                end = j
                break
    body = s[i : end + 1] if end is not None else s[i:]
    for tail in ([""] if end is not None else ["}", "]}", "}]}", '"}', '"}]}', '"]}']):
        txt = re.sub(r",\s*([}\]])", r"\1", body + tail)
        for load in (
            json.loads,
            lambda t: ast.literal_eval(
                re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", re.sub(r"\bnull\b", "None", t)))
            ),
        ):
            try:
                return load(txt)
            except Exception:
                pass
    raise ValueError("the reply is not parseable JSON")


LIMITS = {
    "separate": ("factor", 0.5, 2.0),
    "scale_pcs": ("factor", 0.0, 2.0),
    "whiten": ("strength", 0.0, 0.5),
    "center": ("strength", 0.0, 0.5),
}


def repair(
    spec: Any, L: int, X: np.ndarray, categories: Sequence[str], max_ops: int
) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce a nearly-valid reply into the schema instead of rejecting it: fix the layer field, clip values into
    their ranges, normalise category names, drop operations that cannot be executed. Returns (spec, list of repairs)."""
    notes = []
    if isinstance(spec, list):
        spec = {"ops": spec}
    if not isinstance(spec, dict):
        raise ValueError("reply must be a JSON object")
    if spec.get("layer") != L:
        notes.append(f"layer set to {L}")
        spec["layer"] = L
    spec["preserve_norm"] = bool(spec.get("preserve_norm", True))
    ops = []
    for o in (spec.get("ops") if isinstance(spec.get("ops"), list) else []):
        try:
            key, lo, hi = LIMITS[o["op"]]
            v = float(o[key])
            if o["op"] == "separate":
                o["a"], o["b"] = str(o["a"]).strip().lower(), str(o["b"]).strip().lower()
                assert o["a"] in categories and o["b"] in categories and o["a"] != o["b"]
            if o["op"] == "scale_pcs":
                o["pcs"] = sorted(
                    {
                        int(p)
                        for p in (o["pcs"] if isinstance(o["pcs"], list) else [o["pcs"]])
                        if 0 <= int(p) < len(X) - 1
                    }
                )
                assert o["pcs"]
            if o["op"] == "whiten":
                o["rank"] = int(min(max(int(o["rank"]), 1), 16))
        except Exception:
            notes.append(f"dropped unusable op {json.dumps(o, default=str)[:80]}")
            continue
        if not lo <= v <= hi:
            notes.append(f"{o['op']}.{key} {v} clipped to [{lo}, {hi}]")
        o[key] = min(max(v, lo), hi)
        ops.append(o)
    if len(ops) > max_ops:
        notes.append(f"kept the first {max_ops} of {len(ops)} ops")
        ops = ops[:max_ops]
    spec["ops"] = ops
    return spec, notes


def scaled(spec: Dict[str, Any], a: float) -> Dict[str, Any]:
    """The same spec at strength a (a < 1 weaker, a > 1 stronger): factors move away from / toward 1, strengths scale; both stay inside LIMITS."""
    s = {**spec, "ops": [dict(o) for o in spec["ops"]]}
    for o in s["ops"]:
        key, lo, hi = LIMITS[o["op"]]
        o[key] = round(float(np.clip(1 + a * (float(o[key]) - 1) if key == "factor" else a * float(o[key]), lo, hi)), 4)
    return s


class SvdCache:
    """Remembers the SVD (singular values, right singular vectors) of the layer matrix decomposed last, so the several
    strengths of one proposal share one decomposition. One cache belongs to one target model."""

    def __init__(self) -> None:
        self._store: Dict[Any, Any] = {}

    def get(self, X: np.ndarray, mu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(S, Vt) of X - mu."""
        key = (id(X), X.shape, float(X[0, 0]))
        if key not in self._store:
            self._store.clear()
            self._store[key] = np.linalg.svd(X - mu, full_matrices=False)[1:]
        return self._store[key]


def build_affine(
    spec: Dict[str, Any], X: np.ndarray, cats: np.ndarray, cache: Optional[SvdCache] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compile spec["ops"] into (A, b) such that h' = h @ A + b; all statistics come from X (the layer BEFORE the edit).
    Same semantics as the reference implementation (prompts.REF_IMPL); the SVD is computed only when an operation needs it
    and is cached per layer."""
    cache = SvdCache() if cache is None else cache
    d = X.shape[1]
    mu = X.mean(0)
    A, b = np.eye(d), np.zeros(d)

    def pcs():
        return cache.get(X, mu)

    def push(U, f, c):
        nonlocal A, b
        Mx = np.eye(d) + (U.T * (np.asarray(f, float) - 1)) @ U
        A, b = A @ Mx, b @ Mx + c - c @ Mx

    for o in spec["ops"]:
        if o["op"] == "scale_pcs":
            push(pcs()[1][o["pcs"]], o["factor"], mu)
        elif o["op"] == "whiten":
            S, Vt = pcs()
            s = S[: o["rank"]]
            push(Vt[: o["rank"]], (np.exp(np.log(s).mean()) / s) ** o.get("strength", 1), mu)
        elif o["op"] == "separate":
            ca, cb = X[cats == o["a"]].mean(0), X[cats == o["b"]].mean(0)
            push(((ca - cb) / max(np.linalg.norm(ca - cb), 1e-12))[None], o["factor"], (ca + cb) / 2)
        elif o["op"] == "center":
            b = b - o.get("strength", 1) * (mu @ A + b)
        else:
            raise ValueError("unknown op " + str(o["op"]))
    return A, b


def apply_spec(spec: Dict[str, Any], X: np.ndarray, cats: np.ndarray, cache: Optional[SvdCache] = None) -> np.ndarray:
    """Apply a spec to the point cloud X (simulation: no model pass)."""
    A, b = build_affine(spec, X, cats, cache)
    Y = X @ A + b
    if spec.get("preserve_norm"):
        Y = Y * np.linalg.norm(X, axis=1, keepdims=True) / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def ridge_solve(
    Kx: torch.Tensor, Dx: torch.Tensor, ridge: float, weight: torch.Tensor
) -> Tuple[torch.Tensor, float, float]:
    """Closed-form ridge solution dW with dW @ key ~ target change. Returns (dW, calibration R2, ||dW||/||W||).
    With fewer calibration tokens than key dimensions (the usual case for larger models) the equivalent dual form is solved.
    """
    n, dk = Kx.shape
    eye = lambda k: torch.eye(k, dtype=Kx.dtype, device=Kx.device)
    if n < dk:
        G = Kx @ Kx.T
        dW = (Kx.T @ torch.linalg.solve(G + ridge * G.diagonal().sum() / dk * eye(n), Dx)).T
    else:
        G = Kx.T @ Kx
        dW = torch.linalg.solve(G + ridge * G.diagonal().mean() * eye(dk), Kx.T @ Dx).T
    r2 = 1 - ((Kx @ dW.T - Dx) ** 2).sum() / ((Dx - Dx.mean(0)) ** 2).sum()
    return dW.to(weight.dtype), float(r2), float(dW.norm() / weight.double().norm())
