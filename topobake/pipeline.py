"""The layer loop (sequential and parallel control), the evaluation and summary of one run, and the experiment runner.

Layers are edited one at a time from first to last. For each layer the proposer proposes with measured feedback (proposers.py), the
accepted spec is verified on the real model (verification.py) and compiled into the weights of the block's MLP output projection by
closed-form ridge regression (edits.ridge_solve). If the baked version breaks a limit, the edit is shrunk further; if it still fails,
it is reverted.
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
import torch

from .config import CONDITIONS, MODELS, RULE, SPARE_MODELS, Paths, PipelineConfig, RunConfig
from .data import GEN_CLOZE, GEN_TEMPLATES, Benchmark, ProbeData
from .edits import ridge_solve, scaled
from .geometry import cka, describe, geom, geom_parts, rel_change, topo
from .models import AnalystModel, TargetModel
from .prompts import render_system
from .proposers import HybridProposer, LLMProposer, Proposer, RuleProposer
from .studies import run_id
from .utils import NanReport, cos_shift, fin, free_gpu, kl, pref, ratio
from .verification import limits_check

# Every log row carries every column (NaN control): layers that were not edited get neutral values.
ROW0 = dict(
    status="skipped_quality",
    arm="-",
    analyst="-",
    attempts=0,
    n_ops=0,
    spec="[]",
    source="none",
    rationale="-",
    reject_reason="-",
    scale=0.0,
    sim_gain=0.0,
    spec_rel_change=0.0,
    hooked_step_kl=0.0,
    bake_r2=0.0,
    ridge=0.0,
    rel_weight_change=0.0,
    kl_hook_vs_baked=0.0,
    attempted_step_kl=0.0,
    attempted_cum_kl=0.0,
)

GEO_M = ["knn_purity", "silhouette", "anisotropy", "twonn_id", "betti0_q25", "betti1_q25"]


@dataclass
class RunContext:
    """Everything one run needs, created by ``ExperimentRunner.run_one`` and passed down."""

    cfg: RunConfig
    pcfg: PipelineConfig  # the pipeline constants with this run's strength grid and gain threshold
    data: ProbeData
    bench: Benchmark
    target: TargetModel
    proposer: Proposer
    nan: NanReport
    ripser_fn: Optional[Callable[..., Any]] = None


def pool_entry(name: str) -> Dict[str, Any]:
    """A copy of the pool entry (or spare entry) with this name."""
    for m in list(MODELS) + list(SPARE_MODELS):
        if m["name"] == name:
            return dict(m)
    raise KeyError(f"model '{name}' is not in MODELS / SPARE_MODELS")


def run_dir(root: str, A: Mapping[str, Any], T: Mapping[str, Any]) -> str:
    """Run folder <root>/<arm>/<analyst>/<target>; arms never share a folder."""
    return os.path.join(root, A["arm"], A["name"], T["name"])


# ====================================================================================================
# Sequential loop
# ====================================================================================================
class SequentialRunner:
    """Layer-by-layer loop: each layer is analysed on the already-edited model, verified and baked before the next one."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def process_layer(self, L: int, pre: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Analyse, propose, verify and bake one layer. Returns (log row, model snapshot after this step)."""
        c = self.ctx
        t, data, pcfg = c.target, c.data, c.pcfg
        K, cats = pcfg.k, data.cats
        t0 = time.time()
        X, X0 = pre["H"][:, L], t.h_orig[:, L]
        out = t.layer_dir(L)
        os.makedirs(out, exist_ok=True)
        d = describe(X, L, data, pcfg, c.ripser_fn)
        g = {"orig": geom(X0, cats, K), "before": geom(X, cats, K)}
        d["layer_summary"] = d["layer_summary"].assign(
            **pref(g["orig"], "orig"), drift_rel_l2=rel_change(X0, X), drift_cos=cos_shift(X0, X)
        )
        for k, df in d.items():
            c.nan.clean(df, f"layer/{k}").to_csv(os.path.join(out, f"{k}.csv"), index=False)
        row = {"layer": L, **ROW0}
        post, trace = pre, []

        if kl(t.ref, pre["lp"]) >= pcfg.max_cum_kl or pre["ppl"] / t.base["ppl"] - 1 >= pcfg.max_cum_ppl_increase:
            row["status"] = "skipped_budget"  # the KL or perplexity budget is used up: no further edits
        elif (
            g["before"]["knn_purity"] < pcfg.skip_purity and d["category_pairs"].cross_knn_rate.max() >= pcfg.skip_cross
        ):  # adaptive skipping
            spec, hooked, trace, why = c.proposer.propose(t, L, d, X, pre)
            row.update(status=why, attempts=len(trace))
            if spec:
                m = t.lin(L)
                for a in pcfg.scales:  # shrink further if the BAKED version breaks a limit
                    s = spec if a == pcfg.scales[0] else scaled(spec, a)
                    if a != pcfg.scales[0]:
                        hk = t.add_hook(L, t.make_transform(s, X))
                        hooked = t.snapshot()
                        hk.remove()
                    Kx, Dx = t.bake_data(L, t.make_transform(s, X))
                    for ridge in pcfg.ridges:  # regularisation ladder
                        dW, r2, relw = ridge_solve(Kx, Dx, ridge, m.weight)
                        if relw <= pcfg.max_rel_weight_change:
                            break
                    g["hooked"] = geom(hooked["H"][:, L], cats, K)
                    row.update(
                        n_ops=len(s["ops"]),
                        spec=json.dumps(s["ops"]),
                        rationale=(str(spec.get("rationale", "")).strip() or "-")[:300],
                        source=spec["_sim"]["source"],
                        scale=spec["_sim"]["scale"] * a,
                        sim_gain=spec["_sim"]["gain"],
                        spec_rel_change=spec["_sim"]["rel_change"],
                        hooked_step_kl=kl(pre["lp"], hooked["lp"]),
                        bake_r2=fin(r2),
                        ridge=ridge,
                        rel_weight_change=fin(relw),
                    )
                    if relw > pcfg.max_rel_weight_change:
                        row.update(status="rejected_weight_budget", reject_reason="weight")
                        continue
                    W0 = m.weight.data.clone()
                    m.weight.data.add_(dW)
                    cand = t.snapshot()
                    bad, mm = limits_check(t, pre, cand, L)  # KL, perplexity and CKA limits on the baked model
                    row.update(
                        kl_hook_vs_baked=kl(hooked["lp"], cand["lp"]),
                        attempted_step_kl=mm["step_kl"],
                        attempted_cum_kl=mm["cum_kl"],
                    )
                    if bad is None:
                        cand["hsm"] = t.monitor()  # HellaSwag on the monitor subset
                        if t.base["hsm"] - cand["hsm"] > pcfg.max_hellaswag_drop:
                            bad = "hellaswag"
                    if bad:
                        m.weight.data.copy_(W0)
                        row.update(status="reverted_" + bad, reject_reason=bad)
                        continue
                    row.update(status="applied", reject_reason="-")
                    post = cand
                    t.orig_w[L] = W0
                    torch.save(dW.cpu(), t.delta_path(L))
                    break

        Y = post["H"][:, L]
        g["after"] = geom(Y, cats, K)
        g.setdefault("hooked", g["before"])
        for k, v in g.items():
            row.update(pref(v, k))
        row.update(
            last_layer_knn_purity=geom(post["H"][:, -1], cats, K)["knn_purity"],
            shift_words_rel_l2=rel_change(X, Y),
            shift_heldout_rel_l2=rel_change(pre["hs"][:, L], post["hs"][:, L]),
            step_kl=kl(pre["lp"], post["lp"]),
            cum_kl=kl(t.ref, post["lp"]),
            step_cka_words=cka(X, Y),
            step_cka_tokens=cka(pre["hs"][:, L], post["hs"][:, L]),
            cka_words_to_original=cka(X0, Y),
            heldout_ppl=post["ppl"],
            ppl_change_step=post["ppl"] / pre["ppl"] - 1,
            ppl_change_cum=post["ppl"] / t.base["ppl"] - 1,
            hellaswag_monitor=post["hsm"],
            hellaswag_change_step=post["hsm"] - pre["hsm"],
            cloze_acc=post["cloze"],
            seconds=round(time.time() - t0, 1),
        )
        with open(os.path.join(out, "analyst_log.json"), "w") as f:
            json.dump(trace, f, indent=1, default=float)
        return row, post

    def run(self) -> None:
        """The resumable layer loop; all layers of a run are proposed by the same analyst."""
        c = self.ctx
        t = c.target
        todo = [L for L in t.layers[: c.pcfg.max_layers_per_target] if L not in t.log]
        if not todo:
            return
        cur = t.snapshot()
        cur["hsm"] = t.monitor() if t.orig_w else t.base["hsm"]
        for L in todo:
            row, cur = self.process_layer(L, cur)
            row["analyst"], row["arm"] = c.proposer.analyst_name, c.proposer.arm
            t.log[L] = row
            t.save_state()
            print(
                f"  layer {L:>2} | {row['status']:<22} | {row['source']:<17} | x{row['scale']:<5.2f} | tries {row['attempts']} | "
                f"purity {row['before_knn_purity']:.3f}->{row['after_knn_purity']:.3f} | KL {row['step_kl']:.4f}/{row['cum_kl']:.4f} | "
                f"ppl {row['ppl_change_step']:+.2%}/{row['ppl_change_cum']:+.2%} | CKA {row['step_cka_tokens']:.3f} | "
                f"HS {row['hellaswag_monitor']:.3f} | {row['seconds']:.0f}s"
            )


# ====================================================================================================
# Non-sequential control
# ====================================================================================================
class ParallelRunner:
    """Every layer is analysed on the ORIGINAL model and its edit is fitted on the original model; nothing is applied until all
    layers are done, then all weight deltas are added together. Each edit passes the per-step limits in isolation, but nothing
    controls how the edits interact - which is exactly what the sequential loop adds."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def run(self) -> None:
        c = self.ctx
        t, data, pcfg = c.target, c.data, c.pcfg
        K, cats = pcfg.k, data.cats
        base = t.base
        pending, rows = {}, {}
        for L in t.layers[: pcfg.max_layers_per_target]:
            t0 = time.time()
            X = base["H"][:, L]
            out = t.layer_dir(L)
            os.makedirs(out, exist_ok=True)
            d = describe(X, L, data, pcfg, c.ripser_fn)
            g = geom(X, cats, K)
            d["layer_summary"] = d["layer_summary"].assign(**pref(g, "orig"), drift_rel_l2=0.0, drift_cos=0.0)
            row = {
                "layer": L,
                **ROW0,
                **pref(g, "orig"),
                **pref(g, "before"),
                **pref(g, "hooked"),
                "arm": c.proposer.arm,
                "analyst": c.proposer.analyst_name,
            }
            trace = []
            if g["knn_purity"] < pcfg.skip_purity and d["category_pairs"].cross_knn_rate.max() >= pcfg.skip_cross:
                spec, hooked, trace, why = c.proposer.propose(t, L, d, X, base)
                row.update(status=why, attempts=len(trace))
                if spec:
                    Kx, Dx = t.bake_data(L, t.make_transform(spec, X))
                    for ridge in pcfg.ridges:
                        dW, r2, relw = ridge_solve(Kx, Dx, ridge, t.lin(L).weight)
                        if relw <= pcfg.max_rel_weight_change:
                            break
                    row.update(
                        n_ops=len(spec["ops"]),
                        spec=json.dumps(spec["ops"]),
                        source=spec["_sim"]["source"],
                        scale=spec["_sim"]["scale"],
                        sim_gain=spec["_sim"]["gain"],
                        spec_rel_change=spec["_sim"]["rel_change"],
                        hooked_step_kl=spec["_sim"]["hooked_step_kl"],
                        bake_r2=fin(r2),
                        ridge=ridge,
                        rel_weight_change=fin(relw),
                        **pref(geom(hooked["H"][:, L], cats, K), "hooked"),
                    )
                    if relw > pcfg.max_rel_weight_change:
                        row.update(status="rejected_weight_budget", reject_reason="weight")
                    else:
                        row["status"] = "applied"
                        pending[L] = dW
            with open(os.path.join(out, "analyst_log.json"), "w") as f:
                json.dump(trace, f, indent=1, default=float)
            row["seconds"] = round(time.time() - t0, 1)
            rows[L] = row
            print(
                f"  [parallel] layer {L:>2} | {row['status']:<22} | {row['source']:<17} | x{row['scale']:<5.2f} | planned on the original model"
            )
        for L, dW in pending.items():  # apply everything at once
            m = t.lin(L)
            t.orig_w[L] = m.weight.data.clone()
            m.weight.data.add_(dW)
            torch.save(dW.cpu(), t.delta_path(L))
        post = t.snapshot()
        post["hsm"] = t.monitor() if pending else base["hsm"]
        for L, row in rows.items():  # what each layer looks like once all edits act together
            X, Y = base["H"][:, L], post["H"][:, L]
            row.update(
                pref(geom(Y, cats, K), "after"),
                last_layer_knn_purity=geom(post["H"][:, -1], cats, K)["knn_purity"],
                shift_words_rel_l2=rel_change(X, Y),
                shift_heldout_rel_l2=rel_change(base["hs"][:, L], post["hs"][:, L]),
                step_kl=0.0,
                cum_kl=kl(t.ref, post["lp"]),
                step_cka_words=cka(X, Y),
                step_cka_tokens=cka(base["hs"][:, L], post["hs"][:, L]),
                cka_words_to_original=cka(X, Y),
                heldout_ppl=post["ppl"],
                ppl_change_step=0.0,
                ppl_change_cum=post["ppl"] / base["ppl"] - 1,
                hellaswag_monitor=post["hsm"],
                hellaswag_change_step=0.0,
                cloze_acc=post["cloze"],
            )
            t.log[L] = row
        t.save_state()
        print(
            f"  [parallel] {len(pending)} edits applied together: KL to original {kl(t.ref, post['lp']):.4f}, perplexity {post['ppl'] / base['ppl'] - 1:+.2%}"
        )


# ====================================================================================================
# Evaluation and summary of one run
# ====================================================================================================
def summary_keys(benchmarks: Tuple[str, ...]) -> List[str]:
    """The keys (and their order) of the one-row summary of a run."""
    return (
        [f"{m}_change_{c}" for m in GEO_M for c in CONDITIONS]
        + [
            "transfer_purity_B",
            "transfer_purity_C",
            "transfer_purity_D",
            "cka_probe",
            "cka_unseen",
            "cka_tokens",
            "cum_kl",
            "kl_new_domain",
            "ppl_change_calibration_pct",
            "ppl_change_guard_pct",
            "ppl_change_heldout_pct",
            "ppl_change_new_domain_pct",
            "cloze_change_seen",
            "cloze_change_unseen",
            "fact_acc_change",
            "fact_hallucination_change",
            "fact_margin_change",
        ]
        + [f"{b}_change" for b in benchmarks]
    )


def zero_summary(benchmarks: Tuple[str, ...]) -> Dict[str, float]:
    """No layer was edited: the model is unchanged, so every change is 0 by definition and the before/after evaluation is skipped."""
    s = {k: 0.0 for k in summary_keys(benchmarks)}
    s.update(cka_probe=1.0, cka_unseen=1.0, cka_tokens=1.0)
    return s


def summarize(
    geo: pd.DataFrame,
    cka_df: pd.DataFrame,
    beh: pd.DataFrame,
    edited: List[int],
    log: pd.DataFrame,
    benchmarks: Tuple[str, ...],
) -> Dict[str, float]:
    """The one-row summary of a run: mean change over the edited layers, transfer ratios, CKA, KL, perplexity and benchmark changes."""
    ed = sorted(edited)
    s = {}
    for c in CONDITIONS:
        w = geo[geo.set == c].pivot(index="layer", columns="model")
        for m in GEO_M:
            s[f"{m}_change_{c}"] = fin((w[(m, "edited")] - w[(m, "original")]).loc[ed].mean())
    for c in "BCD":
        s[f"transfer_purity_{c}"] = ratio(s[f"knn_purity_change_{c}"], s["knn_purity_change_A"])
    ck = cka_df[cka_df.edited == 1]
    s.update(
        cka_probe=fin(ck.cka_probe_words.mean(), 1.0),
        cka_unseen=fin(ck.cka_unseen_words.mean(), 1.0),
        cka_tokens=fin(ck.cka_heldout_tokens.mean(), 1.0),
    )
    s["cum_kl"] = fin(log.cum_kl.iloc[-1])
    s["kl_new_domain"] = fin(beh.loc["kl_new_domain", "edited"])
    for k in ("calibration", "guard", "heldout", "new_domain"):
        s[f"ppl_change_{k}_pct"] = 100 * ratio(beh.loc[f"ppl_{k}", "change"], beh.loc[f"ppl_{k}", "original"])
    s.update(
        cloze_change_seen=beh.loc["cloze_seen", "change"],
        cloze_change_unseen=beh.loc["cloze_unseen", "change"],
        fact_acc_change=beh.loc["fact_acc", "change"],
        fact_hallucination_change=beh.loc["fact_hallucination", "change"],
        fact_margin_change=beh.loc["fact_margin", "change"],
    )
    for b in benchmarks:
        s[f"{b}_change"] = float(beh.loc[b, "change"]) if b in beh.index else 0.0
    return {k: fin(s.get(k, 0.0)) for k in summary_keys(benchmarks)}


class RunEvaluator:
    """Original vs edited model on the four conditions A-D plus behaviour; writes the evaluation tables into the run folder."""

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx

    def _eval_model(self, tag: str):
        """Geometry at every layer for the four conditions (global and per category), plus behaviour."""
        c = self.ctx
        t, data, pcfg = c.target, c.data, c.pcfg
        K = pcfg.k
        sets = {
            "A": (data.words, data.cats, [pcfg.template], data.sub),
            "B": (data.gen_words, data.gen_cats, [pcfg.template], data.sub_gen),
            "C": (data.words, data.cats, GEN_TEMPLATES, data.sub),
            "D": (data.gen_words, data.gen_cats, GEN_TEMPLATES, data.sub_gen),
        }
        rows, crows, keep = [], [], {}
        for name, (ws, cs, tpls, sub) in sets.items():
            for f, tp in enumerate(tpls):
                H = t.feats(ws, tp)
                if f == 0:
                    keep[name] = H
                for L in t.layers:
                    a, pur, sil, _, _ = geom_parts(H[:, L], cs, K)
                    rows.append(
                        {
                            "model": tag,
                            "set": name,
                            "frame": f,
                            "layer": L,
                            "anisotropy": float(a),
                            "knn_purity": float(pur.mean()),
                            "silhouette": float(sil.mean()),
                            **topo(H[sub, L], c.ripser_fn),
                        }
                    )
                    crows += [
                        {
                            "model": tag,
                            "set": name,
                            "frame": f,
                            "layer": L,
                            "category": cat,
                            "knn_purity": float(pur[cs == cat].mean()),
                            "silhouette": float(sil[cs == cat].mean()),
                        }
                        for cat in data.categories
                    ]
        geo = pd.DataFrame(rows).drop(columns="frame").groupby(["model", "set", "layer"], as_index=False).mean()
        cat = (
            pd.DataFrame(crows)
            .drop(columns="frame")
            .groupby(["model", "set", "layer", "category"], as_index=False)
            .mean()
        )
        lp_g, hs, ppl_g = t.text_stats(data.guard, True)
        lp_n, _, ppl_n = t.text_stats(data.new_domain)
        beh = {
            "cloze_seen": t.cloze_acc(data.words, data.cats, pcfg.cloze),
            "cloze_unseen": t.cloze_acc(data.gen_words, data.gen_cats, GEN_CLOZE),
            "ppl_calibration": t.text_stats(data.calib)[2],
            "ppl_guard": ppl_g,
            "ppl_heldout": t.text_stats(data.held_eval)[2],
            "ppl_new_domain": ppl_n,
            **{k: t.mc_accuracy(v) for k, v in c.bench.tasks.items() if v},
        }
        f = t.fact_metrics()
        beh.update(
            fact_acc=f["facts: accuracy (true preferred)"],
            fact_hallucination=f["facts: hallucination rate (false preferred)"],
            fact_margin=f["facts: mean truth margin (nats)"],
        )
        return geo, cat, beh, {"H": keep, "hs": hs, "lp_g": lp_g, "lp_n": lp_n}

    def evaluate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Writes eval_layers.csv, per_category_layers.csv, cka.csv, eval_behaviour.csv into the run folder."""
        c = self.ctx
        t, nan = c.target, c.nan
        ge, ce, be, xe = self._eval_model("edited")
        edited = {L: t.lin(L).weight.data.clone() for L in t.orig_w}
        try:
            for L, W in t.orig_w.items():
                t.lin(L).weight.data.copy_(W)
            go_, co, bo, xo = self._eval_model("original")
        finally:
            for L, W in edited.items():
                t.lin(L).weight.data.copy_(W)
        out = t.out_dir
        geo = nan.clean(pd.concat([go_, ge]), "eval_layers")
        geo.to_csv(os.path.join(out, "eval_layers.csv"), index=False)
        cat = nan.clean(pd.concat([co, ce]), "per_category")
        cat.to_csv(os.path.join(out, "per_category_layers.csv"), index=False)
        cka_df = nan.clean(
            pd.DataFrame(
                [
                    {
                        "layer": L,
                        "edited": int(L in t.orig_w),
                        "cka_probe_words": cka(xo["H"]["A"][:, L], xe["H"]["A"][:, L]),
                        "cka_unseen_words": cka(xo["H"]["B"][:, L], xe["H"]["B"][:, L]),
                        "cka_heldout_tokens": cka(xo["hs"][:, L], xe["hs"][:, L]),
                    }
                    for L in t.layers
                ]
            ),
            "cka",
        )
        cka_df.to_csv(os.path.join(out, "cka.csv"), index=False)
        beh = pd.DataFrame({"original": bo, "edited": be})
        beh["change"] = beh.edited - beh.original
        beh.loc["kl_guard"] = [0.0, kl(xo["lp_g"], xe["lp_g"]), kl(xo["lp_g"], xe["lp_g"])]
        beh.loc["kl_new_domain"] = [0.0, kl(xo["lp_n"], xe["lp_n"]), kl(xo["lp_n"], xe["lp_n"])]
        beh = nan.clean(beh, "eval_behaviour")
        beh.index.name = "metric"
        beh.to_csv(os.path.join(out, "eval_behaviour.csv"))
        return geo, cat, cka_df, beh


# ====================================================================================================
# One run
# ====================================================================================================
class ExperimentRunner:
    """Experiment runner. Executes configurations one after another and keeps the analyst, the probe data and the benchmarks in memory between
    runs that share them. A run whose summary.json exists is skipped; an interrupted sequential run resumes at the layer where it stopped.
    """

    def __init__(
        self,
        paths: Paths,
        pcfg: PipelineConfig,
        holdout: float,
        device: str,
        ripser_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.paths, self.pcfg, self.holdout, self.device, self.ripser_fn = paths, pcfg, holdout, device, ripser_fn
        self.nan = NanReport()
        self._analyst: Optional[AnalystModel] = None
        self._analyst_name: Optional[str] = None
        self._data: Optional[ProbeData] = None
        self._data_key: Optional[Tuple[int, int]] = None
        self._bench: Optional[Benchmark] = None
        self._bench_seed: Optional[int] = None

    # ---- cached resources ----
    def _get_data(self, n_categories: int, seed: int) -> ProbeData:
        """The probe data of this category count and seed (re-read only when one of them changed)."""
        if self._data_key != (n_categories, seed):
            self._data = ProbeData.load(self.paths.data_dir, n_categories, seed, self.holdout, self.pcfg)
            self._data_key = (n_categories, seed)
        return self._data

    def _get_bench(self, seed: int) -> Benchmark:
        """The benchmark subsets of this seed (re-drawn only when the seed changed)."""
        if self._bench_seed != seed:
            self._bench = Benchmark.load(seed, self.paths.cache_dir, self.pcfg)
            self._bench_seed = seed
        return self._bench

    def _unload_analyst(self) -> None:
        if self._analyst is not None:
            self._analyst.unload()
        self._analyst = None
        self._analyst_name = None
        free_gpu()

    def _set_analyst(self, cfg: RunConfig) -> Dict[str, Any]:
        """Load the analyst of this run if it is not the one already in memory. Returns its pool entry (with the arm)."""
        A = dict(RULE) if cfg.arm == "rule" else dict(pool_entry(cfg.analyst), arm=cfg.arm)
        if self._analyst_name != A["name"]:
            self._unload_analyst()  # the arm decides which kinds of proposal may be applied in this round
            if A["repo"] == "heuristic":
                print(f"[round] arm: {A['arm']} | proposer: rules only")
            else:
                self._analyst = AnalystModel(A, self.paths, self.pcfg, self.device).load()
            self._analyst_name = A["name"]
        return A

    def close(self) -> None:
        """Release the analyst."""
        self._unload_analyst()

    # ---- one run ----
    def run_one(self, cfg: Union[RunConfig, Mapping[str, Any]]) -> Dict[str, Any]:
        """Run (or load from cache) the configuration cfg and return its one-row summary."""
        cfg = cfg if isinstance(cfg, RunConfig) else RunConfig.from_mapping(cfg)
        c = cfg.to_dict()
        rid = run_id(cfg)
        rdir = os.path.join(self.paths.runs_dir, rid)
        done = os.path.join(rdir, "summary.json")
        if os.path.exists(done):
            print(f"[cached] {rid}")
            with open(done) as f:
                return json.load(f)
        print(f"\n===== RUN {rid} =====")
        t0 = time.time()
        data, bench = self._get_data(cfg.n_categories, cfg.seed), self._get_bench(cfg.seed)
        pcfg = replace(
            self.pcfg,
            adapt_grid=self.pcfg.adapt_grid if cfg.alpha == "adaptive" else (float(cfg.alpha),),
            min_gain=float(cfg.min_gain),
        )
        system = render_system(pcfg, len(data.words), data.categories)
        A = self._set_analyst(cfg)
        T = pool_entry(cfg.target)
        proposer_cls = {"llm": LLMProposer, "rule": RuleProposer, "hybrid": HybridProposer}[A["arm"]]
        proposer = proposer_cls(
            self._analyst, A["name"], pcfg, system, descriptors=cfg.descriptors, relaxed=cfg.relaxed
        )
        out_dir = run_dir(rdir, A, T)
        if cfg.mode == "parallel":  # the parallel control has no partial resume: start clean
            shutil.rmtree(out_dir, ignore_errors=True)
        target = TargetModel(
            T,
            out_dir,
            os.path.join(self.paths.edited_dir, f"{T['name']}-{A['arm']}-{A['name']}"),
            self.paths,
            pcfg,
            data,
            bench,
            self.nan,
            self.device,
        )
        target.load(A["arm"], A["name"])
        ctx = RunContext(
            cfg=cfg,
            pcfg=pcfg,
            data=data,
            bench=bench,
            target=target,
            proposer=proposer,
            nan=self.nan,
            ripser_fn=self.ripser_fn,
        )
        try:
            (SequentialRunner if cfg.mode == "sequential" else ParallelRunner)(ctx).run()
            log = self.nan.clean(pd.DataFrame([target.log[L] for L in sorted(target.log)]), "pipeline_log")
            ap = log[log.status == "applied"]
            meta = {
                **c,
                "run_id": rid,
                "run_dir": target.out_dir,
                "n_layers": len(target.layers),
                "layers_processed": len(log),
                "edited_layers": len(ap),
                "edited_layer_list": sorted(target.orig_w),
                "edited_pct": 100 * len(ap) / max(len(log), 1),
                "n_llm_edits": int(ap.source.isin(["llm", "llm_relaxed"]).sum()),
                "n_rule_edits": int(ap.source.isin(["heuristic", "heuristic_relaxed"]).sum()),
                "n_relaxed_edits": int(ap.source.str.endswith("_relaxed").sum()),
                "mean_strength": fin(ap.scale.mean()),
                "mean_step_kl": fin(ap.step_kl.mean()),
                "mean_purity_gain_per_edit": fin((ap.after_knn_purity - ap.before_knn_purity).mean()),
                "n_probe_words": len(data.words),
                "n_unseen_words": len(data.gen_words),
                "original_ppl": target.base["ppl"],
                "original_cloze": target.base["cloze"],
            }
            if not target.orig_w:
                print(
                    "  no layer was edited: the model is unchanged, before/after evaluation skipped, all changes recorded as 0"
                )
                s = zero_summary(pcfg.benchmarks)
                meta["unchanged"] = 1
            else:
                print(f"  evaluating original vs edited model ({len(target.orig_w)} edited layers) ...")
                geo, cat, cka_df, beh = RunEvaluator(ctx).evaluate()
                s = summarize(
                    geo,
                    cka_df,
                    beh,
                    list(target.orig_w),
                    pd.DataFrame([target.log[L] for L in sorted(target.log)]),
                    pcfg.benchmarks,
                )
                meta["unchanged"] = 0
            out = {**meta, **s, "minutes": round((time.time() - t0) / 60, 1)}
            with open(done, "w") as f:
                json.dump(out, f, indent=1, default=float)
            print(
                f"  done in {out['minutes']} min | edited layers {out['edited_layers']}/{out['n_layers']} | purity change A {out['knn_purity_change_A']:+.4f}, "
                f"D {out['knn_purity_change_D']:+.4f} | cum KL {out['cum_kl']:.4f} | ppl new-domain {out['ppl_change_new_domain_pct']:+.2f}%"
            )
            return out
        finally:
            target.unload()
