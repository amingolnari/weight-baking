"""Verification of one proposal: behaviour-preservation limits, adaptive-strength test on the real model, measured feedback.

For a layer the active proposer proposes with measured feedback; proposals that are too strong are shrunk. Safety limits are never
relaxed. Outcomes of a layer: `skipped_quality`, `skipped_by_analyst`, `no_safe_edit`, `rejected_weight_budget`, `reverted_step_kl`,
`halted_cum_kl`, `applied` (see pipeline.py for the layer loop).
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import PipelineConfig
from .edits import apply_spec, scaled
from .geometry import cka, geom, rel_change
from .models import TargetModel
from .utils import kl


def limits_check(
    target: TargetModel, pre: Dict[str, Any], cand: Dict[str, Any], L: int
) -> Tuple[Optional[str], Dict[str, float]]:
    """All behaviour-preservation limits for a candidate model state (hooked or baked), measured against the state before this step
    and against the original model. Returns (name of the first violated limit or None, measurements)."""
    pcfg = target.pcfg
    m = {
        "step_kl": kl(pre["lp"], cand["lp"]),
        "cum_kl": kl(target.ref, cand["lp"]),
        "ppl_step": cand["ppl"] / pre["ppl"] - 1,
        "ppl_cum": cand["ppl"] / target.base["ppl"] - 1,  # perplexity preservation
        "cka_words": cka(pre["H"][:, L], cand["H"][:, L]),
        "cka_tokens": cka(pre["hs"][:, L], cand["hs"][:, L]),
    }  # is the layer still "itself"?
    why = (
        "kl"
        if m["step_kl"] > pcfg.max_step_kl
        else (
            "cum_kl"
            if m["cum_kl"] > pcfg.max_cum_kl
            else (
                "ppl"
                if m["ppl_step"] > pcfg.max_step_ppl_increase
                else (
                    "cum_ppl"
                    if m["ppl_cum"] > pcfg.max_cum_ppl_increase
                    else "cka" if min(m["cka_words"], m["cka_tokens"]) < pcfg.min_step_cka else None
                )
            )
        )
    )
    return why, m


def test_spec(
    target: TargetModel,
    spec: Dict[str, Any],
    L: int,
    X: np.ndarray,
    pre: Dict[str, Any],
    g0: Dict[str, float],
    min_gain: float,
    rec: Dict[str, Any],
    weak: List[Tuple[float, Dict[str, Any]]],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Verify one proposal with ADAPTIVE strength.
    1. The proposal is simulated at every strength in the adaptive grid (cheap: no model pass): gain, relative change, CKA.
    2. Strengths inside the size and CKA limits with enough gain are ordered by gain.
    3. They are tested on the real model through the hook, strongest gain first; after a failure only weaker strengths remain.
    Returns (accepted spec at its chosen strength, hooked snapshot) or (None, None)."""
    pcfg, cats, K = target.pcfg, target.data.cats, target.pcfg.k
    cands, seen = [], set()
    for a in pcfg.adapt_grid:
        s = scaled(spec, a)
        key = json.dumps(s["ops"], sort_keys=True)
        if key in seen:
            continue  # clipping can make two strengths identical
        seen.add(key)
        Y = apply_spec(s, X, cats, target.svd)
        g1 = geom(Y, cats, K)
        cands.append(
            {
                "a": a,
                "s": s,
                "rel": rel_change(X, Y),
                "g1": g1,
                "cka": cka(X, Y),
                "gain": g1["knn_purity"] - g0["knn_purity"] + g1["silhouette"] - g0["silhouette"],
            }
        )
    ref = min(cands, key=lambda c: abs(c["a"] - 1.0))  # what the proposal does as written
    rec.update(
        sim=ref["g1"],
        rel_change=ref["rel"],
        gain=ref["gain"],
        sim_cka=ref["cka"],
        strength_scan=[
            {
                "strength": c["a"],
                "gain": round(c["gain"], 4),
                "rel_change": round(c["rel"], 4),
                "cka": round(c["cka"], 4),
            }
            for c in cands
        ],
    )
    ok = [c for c in cands if c["rel"] <= pcfg.max_rel_change and c["cka"] >= pcfg.min_step_cka]
    if not ok:
        rec["reason"] = "size"
        return None, None
    good = sorted([c for c in ok if c["gain"] >= min_gain], key=lambda c: -c["gain"])
    if not good:
        rec["reason"] = "gain"
        best = max(ok, key=lambda c: c["gain"])
        if best["gain"] > 0:
            weak.append((best["gain"], best["s"]))
        return None, None
    tests = 0
    while good and tests < pcfg.max_hook_tests:
        c = good.pop(0)
        tests += 1
        hk = target.add_hook(L, target.make_transform(c["s"], X))
        hooked = target.snapshot()
        hk.remove()  # behavioural test, weights untouched
        why, m = limits_check(target, pre, hooked, L)
        rec.update(hooked=m, tested_strength=c["a"])
        if why is None:
            c["s"]["_sim"] = {
                "gain": c["gain"],
                "rel_change": c["rel"],
                "hooked_step_kl": m["step_kl"],
                "scale": c["a"],
            }
            rec["verdict"] = f"accepted at strength {c['a']}"
            rec.pop("reason", None)
            return c["s"], hooked
        rec["reason"] = why
        good = [g_ for g_ in good if g_["a"] < c["a"]]  # only weaker versions can still pass
    return None, None


def feedback(rec: Dict[str, Any], L: int, g0: Dict[str, float], pcfg: PipelineConfig) -> str:
    """The measured numbers of a failed proposal, phrased as the message returned to the analyst."""
    g1 = rec["sim"]
    h = rec.get("hooked")
    meas = (
        f"Simulated on layer {L} as written: knn_purity {g0['knn_purity']:.3f}->{g1['knn_purity']:.3f}, silhouette {g0['silhouette']:.3f}->{g1['silhouette']:.3f}, "
        f"anisotropy {g0['anisotropy']:.3f}->{g1['anisotropy']:.3f}, relative change {rec['rel_change']:.3f}, CKA {rec['sim_cka']:.3f}."
    )
    if h:
        meas += (
            f" On the model at strength {rec['tested_strength']}: KL {h['step_kl']:.4f} (limit {pcfg.max_step_kl}), perplexity {h['ppl_step']:+.1%} for this edit "
            f"(limit {pcfg.max_step_ppl_increase:+.0%}) and {h['ppl_cum']:+.1%} in total (limit {pcfg.max_cum_ppl_increase:+.0%}), token CKA {h['cka_tokens']:.3f} (minimum {pcfg.min_step_cka})."
        )
    head = {
        "size": f"REJECTED: at every strength the edit is too large (relative change above {pcfg.max_rel_change} or CKA below {pcfg.min_step_cka}). Use fewer operations.",
        "gain": f"REJECTED: the edit does not improve the layer at any strength (gain {rec['gain']:+.3f}, need {pcfg.min_gain}). Target different category pairs or a different operation.",
        "kl": "REJECTED: geometry improved but the model's predictions changed too much, even at reduced strength. Propose a different, gentler edit.",
        "cum_kl": "REJECTED: the total divergence from the original model would exceed its limit. Propose a much gentler edit or return no edit.",
        "ppl": "REJECTED: geometry improved but perplexity rose too much, even at reduced strength. Propose an edit that disturbs fewer directions.",
        "cum_ppl": "REJECTED: total perplexity would exceed its limit against the original model. Propose a much gentler edit or return no edit.",
        "cka": "REJECTED: the edit changes what the layer represents too much (CKA too low on real text). Propose a more local edit.",
    }[rec["reason"]]
    return head + " " + meas
