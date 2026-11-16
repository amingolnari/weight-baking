"""Analysis: tables, statistics, figures and ABLATION_SUMMARY.md.

One observation is one run (a target x seed, for one analyst). For every variant and metric the tables give the mean, the standard
deviation, a 95% t confidence interval, the p-value of a one-sample t-test against 0, and - against the study's reference variant -
the mean paired difference and the p-value of a paired t-test over runs matched on analyst, target and seed. Seeds re-draw the
probe/unseen split and the benchmark samples; they are not independent model trainings, so the intervals describe the variability
of data and evaluation. With fewer than two observations the interval collapses to the mean and p is reported as 1.
"""

import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as st


def _pyplot():
    """matplotlib.pyplot with the non-interactive backend (selected on first use, not at import time)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


HEAD = [
    "edited_layers",
    "knn_purity_change_A",
    "knn_purity_change_D",
    "silhouette_change_D",
    "cka_unseen",
    "cum_kl",
    "ppl_change_heldout_pct",
    "ppl_change_new_domain_pct",
    "cloze_change_unseen",
    "hellaswag_change",
]
LABEL = {
    "edited_layers": "accepted edits (layers)",
    "knn_purity_change_A": "purity gain, probe words (A)",
    "knn_purity_change_B": "purity gain, unseen words (B)",
    "knn_purity_change_C": "purity gain, unseen frames (C)",
    "knn_purity_change_D": "purity gain, unseen words+frames (D)",
    "silhouette_change_D": "silhouette gain (D)",
    "cka_unseen": "CKA to original (unseen words)",
    "cum_kl": "cumulative KL",
    "ppl_change_heldout_pct": "perplexity change %, held-out",
    "ppl_change_new_domain_pct": "perplexity change %, new domain",
    "cloze_change_unseen": "cloze accuracy change, unseen",
    "hellaswag_change": "HellaSwag accuracy change",
}
STUDY = {  # study -> (title, what it isolates, setup, columns that define a variant)
    "baseline": (
        "Repeated runs (statistical testing)",
        "Whether the changes produced by the default pipeline are distinguishable from zero.",
        "Default configuration (sequential, adaptive strength, full descriptors), repeated with different seeds.",
        ["arm"],
    ),
    "non_sequential": (
        "Non-sequential control",
        "The benefit of analysing each layer on the already-edited model.",
        "Sequential loop versus a parallel control in which every layer is analysed and fitted on the original model and all edits are applied together.",
        ["arm", "variant"],
    ),
    "analyst_type": (
        "Analyst type",
        "Who should write the edits.",
        "LLM-only, rule-only and hybrid (LLM first, rules when the LLM fails), each as a separate set of edited models.",
        ["arm"],
    ),
    "descriptors": (
        "Descriptor ablation",
        "Which descriptors the analyst LLM actually needs.",
        "The analyst's input is restricted to purity+silhouette, then anisotropy is added, then intrinsic dimension and Betti numbers; 'full' also shows drift and the intra/inter ratio.",
        ["arm", "variant"],
    ),
    "strength": (
        "Strength ablation",
        "The effect of the strength scaling factor alpha.",
        "One fixed alpha per variant instead of the adaptive search (factors are clipped to the operation limits, so alpha = 2 saturates).",
        ["arm", "variant"],
    ),
    "gain_threshold": (
        "Gain-threshold ablation",
        "How strict the acceptance threshold should be.",
        "Minimum simulated gain in purity + silhouette; the relaxed-threshold stage is switched off so that the threshold binds.",
        ["arm", "variant"],
    ),
    "categories": (
        "Category-count ablation",
        "Whether the method scales with the granularity of the probe set.",
        "The first 4, 8 or 16 categories of probe_words.txt.",
        ["arm", "variant"],
    ),
}
DATA_FILES = ("probe_words.txt", "calibration_sentences.txt", "heldout_sentences.txt", "new_domain_sentences.txt")


def load_runs(results_dir: str) -> pd.DataFrame:
    """The summary.json of every saved run as one table (empty when there is none)."""
    rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "summary.json")))]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def run_folder(run: Dict[str, Any], results_dir: str) -> str:
    """The folder of a run: the recorded one, or (when the results were moved) the same location under ``results_dir``."""
    d = run.get("run_dir")
    if d and os.path.isdir(d):
        return d
    return os.path.join(results_dir, "runs", run["run_id"], run["arm"], run["analyst"], run["target"])


def stat_summary(x) -> Dict[str, float]:
    """n, mean, sd, 95% CI and p-value against 0 for one sample of numbers."""
    x = np.asarray(x, float)
    n = len(x)
    m = float(x.mean()) if n else 0.0
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        h = st.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        p = float(st.ttest_1samp(x, 0.0).pvalue)
    else:
        h, p = 0.0, 1.0
    return {"n": n, "mean": m, "sd": sd, "ci_low": m - h, "ci_high": m + h, "p_vs_zero": p}


def paired(a: pd.Series, b: pd.Series):
    """Mean paired difference a - b and its p-value, for two series indexed by the same run keys."""
    k = a.index.intersection(b.index)
    d = (a.loc[k] - b.loc[k]).values.astype(float)
    if len(d) > 1 and d.std(ddof=1) > 0:
        return float(d.mean()), float(st.ttest_rel(a.loc[k], b.loc[k]).pvalue), len(d)
    return (float(d.mean()) if len(d) else 0.0), 1.0, len(d)


def study_table(S: pd.DataFrame, by: List[str], metrics: List[str], compare: bool = True) -> pd.DataFrame:
    """Long table: one row per variant x metric, with statistics and the paired comparison against the reference variant.
    The reference is the first variant of the same arm (or the first arm when the arms themselves are the variants)."""
    groups = [(nm if isinstance(nm, tuple) else (nm,), g) for nm, g in S.groupby(by, sort=False)]
    out = []
    key = ["target", "seed"] if by == ["arm"] else ["analyst", "target", "seed"]  # arms are matched on target and seed
    for name, g in groups:
        ref = (groups[0][1] if by == ["arm"] else next(r for nm, r in groups if nm[0] == name[0])) if compare else g
        for m in metrics:
            d, p, k = paired(g.groupby(key)[m].mean(), ref.groupby(key)[m].mean()) if ref is not g else (0.0, 1.0, 0)
            out.append(
                {
                    **dict(zip(by, name)),
                    "metric": m,
                    **stat_summary(g[m]),
                    "diff_vs_reference": d,
                    "p_vs_reference": p,
                    "n_paired": k,
                }
            )
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    """A pipe table in Markdown; float columns are rounded to ``digits`` decimals."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: f"{v:.{digits}f}")
    return "\n".join(
        ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
        + ["| " + " | ".join(map(str, r)) + " |" for r in df.values]
    )


def wide(T: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """Compact view of a study table: mean [CI] per variant for the headline metrics."""
    T = T[T.metric.isin(HEAD)].copy()
    T["value"] = [f"{r.mean:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}]" for r in T.itertuples()]
    T["variant"] = T[by].astype(str).agg(" / ".join, axis=1)
    T["metric"] = T.metric.map(LABEL)
    return (
        T.pivot(index="variant", columns="metric", values="value")
        .reindex(columns=[LABEL[m] for m in HEAD if LABEL[m] in set(T.metric)])
        .reset_index()
    )


def findings(T: pd.DataFrame, by: List[str]) -> List[str]:
    """Plain statements read off the table, with their numbers. No interpretation beyond the direction and the test."""
    out = []
    T = T.copy()
    T["variant"] = T[by].astype(str).agg(" / ".join, axis=1)
    for m in ("edited_layers", "knn_purity_change_D", "cum_kl", "ppl_change_new_domain_pct"):
        q = T[T.metric == m]
        if q.empty:
            continue
        best = q.loc[q["mean"].idxmax()]
        worst = q.loc[q["mean"].idxmin()]
        if abs(best["mean"] - worst["mean"]) < 1e-12:
            out.append(f"- **{LABEL[m]}**: the same for every variant ({best['mean']:.4f}).")
            continue
        out.append(
            f"- **{LABEL[m]}**: highest for `{best.variant}` ({best['mean']:.4f}, 95% CI {best.ci_low:.4f} to {best.ci_high:.4f}, n = {int(best.n)}), "
            f"lowest for `{worst.variant}` ({worst['mean']:.4f})."
        )
        sig = q[(q.n_paired > 1) & (q.p_vs_reference < 0.05)]
        for r in sig.itertuples():
            out.append(
                f"  - `{r.variant}` differs from its reference by {r.diff_vs_reference:+.4f} (paired t-test p = {r.p_vs_reference:.3g}, {int(r.n_paired)} matched runs)."
            )
    q = T[(T.metric == "knn_purity_change_D")]
    for r in q.itertuples():
        verdict = "distinguishable from zero" if r.p_vs_zero < 0.05 else "not distinguishable from zero"
        out.append(
            f"- Purity gain on unseen words and frames for `{r.variant}`: {r.mean:+.4f}, {verdict} (p = {r.p_vs_zero:.3g}, n = {int(r.n)})."
        )
    return out


def bars(T: pd.DataFrame, by: List[str], title: str, path: str) -> None:
    """Bar chart (mean and 95% CI over runs) of the headline metrics of a study."""
    plt = _pyplot()
    ms = [
        m for m in ("edited_layers", "knn_purity_change_D", "cum_kl", "ppl_change_new_domain_pct") if m in set(T.metric)
    ]
    fig, axes = plt.subplots(1, len(ms), figsize=(4.6 * len(ms), 4.2), squeeze=False)
    T = T.copy()
    T["variant"] = T[by].astype(str).agg(" / ".join, axis=1)
    for ax, m in zip(axes[0], ms):
        q = T[T.metric == m]
        x = np.arange(len(q))
        ax.bar(x, q["mean"], yerr=[q["mean"] - q.ci_low, q.ci_high - q["mean"]], capsize=3, color="#4c78a8")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(q.variant, rotation=40, ha="right", fontsize=8)
        ax.set_title(LABEL[m], fontsize=10)
    fig.suptitle(f"{title} (mean and 95% CI over runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trajectory(run: Dict[str, Any], path: str, results_dir: str) -> bool:
    """Layer-by-layer view of one run: purity, silhouette, anisotropy (original vs edited), CKA and cumulative KL."""
    d = run_folder(run, results_dir)
    try:
        geo, cka, log = (
            pd.read_csv(os.path.join(d, "eval_layers.csv")),
            pd.read_csv(os.path.join(d, "cka.csv")),
            pd.read_csv(os.path.join(d, "pipeline_log.csv"), keep_default_na=False),
        )
    except FileNotFoundError:
        return False
    plt = _pyplot()
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.2))
    col = {"A": "#1f77b4", "D": "#d62728"}
    for ax, m in zip(axes[:3], ("knn_purity", "silhouette", "anisotropy")):
        for c, name in (("A", "probe words"), ("D", "unseen words + frames")):
            w = geo[geo.set == c].pivot(index="layer", columns="model", values=m)
            ax.plot(w.index, w["original"], ls=":", color=col[c], label=f"{name}, original")
            ax.plot(w.index, w["edited"], color=col[c], label=f"{name}, edited")
        for L in run.get("edited_layer_list", []):
            ax.axvline(L, color="grey", lw=0.3, alpha=0.5)
        ax.set_title(m)
        ax.set_xlabel("layer")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].legend(fontsize=7)
    for c, name in (
        ("cka_probe_words", "probe words"),
        ("cka_unseen_words", "unseen words"),
        ("cka_heldout_tokens", "held-out tokens"),
    ):
        axes[3].plot(cka.layer, cka[c], label=name)
    axes[3].set_title("CKA, original vs edited")
    axes[3].set_xlabel("layer")
    axes[3].legend(fontsize=7)
    axes[4].plot(log.layer, log.cum_kl.astype(float), color="black", label="cumulative KL")
    axes[4].bar(log.layer, log.step_kl.astype(float), color="#ff7f0e", alpha=0.6, label="KL of the step")
    axes[4].set_title("KL to the original model")
    axes[4].set_xlabel("layer")
    axes[4].legend(fontsize=7)
    fig.suptitle(
        f"{run['arm']} | analyst {run['analyst']} -> target {run['target']} | {run['mode']} | seed {run['seed']} (grey lines: edited layers)"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


class AblationReport:
    """Writes the result tables (CSV), the figures and ABLATION_SUMMARY.md from the saved runs of a results folder."""

    def __init__(
        self, results_dir: str, plots_dir: str, summary_path: str, data_dir: str, main_results: Optional[str] = None
    ) -> None:
        self.results_dir, self.plots_dir, self.summary_path, self.data_dir, self.main_results = (
            results_dir,
            plots_dir,
            summary_path,
            data_dir,
            main_results,
        )
        self.md: List[str] = []
        self.R = pd.DataFrame()
        self.base = pd.DataFrame()  # default-configuration runs: baseline and analyst-type studies
        self.E = pd.DataFrame()  # runs of the analyst-efficiency and transfer analysis

    def _csv(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(os.path.join(self.results_dir, name), index=False)

    @staticmethod
    def _not_run(study: str) -> str:
        return f"_Not run yet._ Run `python scripts/run_ablation.py --ablation-type {'repeats' if study == 'baseline' else study}`."

    def write(self) -> None:
        """Rebuild every table, figure and the summary file."""
        os.makedirs(self.plots_dir, exist_ok=True)
        self.R = load_runs(self.results_dir)
        self._header()
        n_runs = len(self.R)
        if n_runs:
            self._studies()
            self._trajectories()
            self._categories()
        else:
            self._no_runs()
        self._efficiency_and_transfer()
        os.makedirs(os.path.dirname(self.summary_path) or ".", exist_ok=True)
        with open(self.summary_path, "w", encoding="utf8") as f:
            f.write("\n".join(self.md) + "\n")
        print(
            f"analysis written: {n_runs} ablation runs, {len(self.E)} runs in the efficiency/transfer analysis -> {self.results_dir}, {self.plots_dir}, {self.summary_path}"
        )

    def _header(self) -> None:
        md = self.md
        placeholder = [
            f
            for f in DATA_FILES
            if os.path.exists(os.path.join(self.data_dir, f))
            and "PLACEHOLDER" in open(os.path.join(self.data_dir, f), encoding="utf8").readline()
        ]
        md += [
            "# Ablation summary",
            "",
            "Generated by `scripts/run_ablation.py` from the runs in `results/`. It is rewritten at the end of every execution (`--analyze-only` rebuilds it without running models).",
            "",
        ]
        if placeholder:
            md += [
                f"> **Placeholder data files detected:** {', '.join(placeholder)}. Ablation runs made with them use template sentences and the short built-in word lists "
                "and are not paper results. Rows taken from the main study used that script's own data.",
                "",
            ]
        md += [
            "## How to read the tables",
            "",
            "Each cell is the mean over runs with its 95% confidence interval. A run is one target and seed for one analyst. Geometry changes are means over the edited layers, "
            "edited minus original. Conditions: A probe words in the probe frame (in-sample), B unseen words, C unseen sentence frames, D unseen words in unseen frames. "
            "A run in which no layer was edited is recorded with change 0 for every metric and its before/after evaluation is skipped.",
            "",
        ]

    def _studies(self) -> None:
        """One section per study with its table, figure and findings."""
        md, R = self.md, self.R
        n_runs = len(R)
        mpath = os.path.join(self.results_dir, "manifest.csv")
        num = R.select_dtypes(include=[np.number]).columns
        R[num] = R[num].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._csv(R.drop(columns=["edited_layer_list"], errors="ignore"), "all_runs.csv")
        metrics = [
            c
            for c in R.columns
            if c.endswith(
                (
                    "_change_A",
                    "_change_B",
                    "_change_C",
                    "_change_D",
                    "_pct",
                    "_change",
                    "_change_seen",
                    "_change_unseen",
                )
            )
            or c
            in (
                "edited_layers",
                "edited_pct",
                "n_relaxed_edits",
                "mean_strength",
                "mean_step_kl",
                "mean_purity_gain_per_edit",
                "cka_probe",
                "cka_unseen",
                "cka_tokens",
                "cum_kl",
                "kl_new_domain",
                "transfer_purity_B",
                "transfer_purity_C",
                "transfer_purity_D",
                "minutes",
            )
        ]
        Mf = (
            pd.read_csv(mpath, keep_default_na=False)[["study", "variant", "run_id"]]
            if os.path.exists(mpath)
            else pd.DataFrame({"study": "baseline", "variant": "default", "run_id": R.run_id})
        )
        J = Mf.merge(R, on="run_id")
        J.loc[J.study == "repeats", "study"] = "baseline"
        J = J.drop_duplicates(["study", "variant", "run_id"])
        md += [
            f"Runs available: {n_runs} ({int(R.unchanged.sum())} of them left the model unchanged). Models: analysts {sorted(set(R.analyst))}, targets {sorted(set(R.target))}.",
            "",
        ]
        for study, (title, isolates, setup, by) in STUDY.items():
            S = J[J.study == study]
            md += [f"## {title}", "", f"**Purpose.** {isolates}", "", f"**Setup.** {setup}", ""]
            if S.empty:
                md += [self._not_run(study), ""]
                continue
            T = study_table(S, by, metrics)
            fname = "repeated_runs_stats.csv" if study == "baseline" else f"ablation_{study}.csv"
            self._csv(T, fname)
            bars(T, by, title, os.path.join(self.plots_dir, f"ablation_{study}.png"))
            md += [
                f"Runs: {len(S)} (seeds {sorted(set(S.seed))}). Full statistics: `results/{fname}`; figure: `plots/ablation_{study}.png`.",
                "",
                md_table(wide(T, by)),
                "",
                "**Findings (read off the table).**",
                "",
                *findings(T, by),
                "",
            ]
            if study == "baseline":  # per (arm, analyst, target) statistics over seeds
                per = study_table(S, ["arm", "analyst", "target"], HEAD, compare=False)
                self._csv(per, "repeated_runs_by_pair.csv")
                md += ["Per analyst-target pair over seeds: `results/repeated_runs_by_pair.csv`.", ""]
        self.base = J[J.study.isin(["baseline", "analyst_type"])].drop_duplicates("run_id")

    def _trajectories(self) -> None:
        """Per-layer trajectories (default-configuration runs that changed the model)."""
        base = self.base
        made = [
            r["run_id"]
            for r in base[base.unchanged == 0].to_dict("records")
            if trajectory(r, os.path.join(self.plots_dir, f"trajectory_{r['run_id']}.png"), self.results_dir)
        ]
        self.md += [
            "## Per-layer trajectories",
            "",
            f"{len(made)} figures `plots/trajectory_<run id>.png`: kNN purity, silhouette and anisotropy by layer for the original (dotted) and edited (solid) model "
            "on probe words and on unseen words in unseen frames, CKA between the two models by layer, and KL to the original model by layer.",
            "",
        ]

    def _categories(self) -> None:
        """Category-level breakdown of the purity and silhouette gains."""
        md, base = self.md, self.base
        plt = _pyplot()
        rows = []
        for r in base[base.unchanged == 0].to_dict("records"):
            p = os.path.join(run_folder(r, self.results_dir), "per_category_layers.csv")
            if not os.path.exists(p) or not r.get("edited_layer_list"):
                continue
            c = pd.read_csv(p)
            c = (
                c[c.layer.isin(r["edited_layer_list"])]
                .groupby(["set", "category", "model"])[["knn_purity", "silhouette"]]
                .mean()
                .unstack("model")
            )
            for (s_, cat), v in c.iterrows():
                if s_ in ("A", "D"):
                    rows.append(
                        {
                            "arm": r["arm"],
                            "run_id": r["run_id"],
                            "condition": s_,
                            "category": cat,
                            "purity_gain": v[("knn_purity", "edited")] - v[("knn_purity", "original")],
                            "silhouette_gain": v[("silhouette", "edited")] - v[("silhouette", "original")],
                        }
                    )
        md += ["## Category-level breakdown", ""]
        if rows:
            C = pd.DataFrame(rows)
            G = (
                C.groupby(["arm", "condition", "category"])
                .agg(
                    purity_gain=("purity_gain", "mean"),
                    silhouette_gain=("silhouette_gain", "mean"),
                    n_runs=("run_id", "nunique"),
                )
                .reset_index()
            )
            self._csv(G, "per_category_gains.csv")
            arms = list(dict.fromkeys(G.arm))
            fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
            for ax, m in zip(axes, ("purity_gain", "silhouette_gain")):
                q = G[G.condition == "D"]
                cats = list(dict.fromkeys(q.category))
                x = np.arange(len(cats))
                w = 0.8 / len(arms)
                for i, a in enumerate(arms):
                    ax.bar(x + i * w, q[q.arm == a].set_index("category").reindex(cats)[m].fillna(0), w, label=a)
                ax.set_xticks(x + 0.4 - w / 2)
                ax.set_xticklabels(cats, rotation=40, ha="right")
                ax.axhline(0, color="black", lw=0.6)
                ax.set_title(f"{m.replace('_', ' ')} per category, unseen words + frames (D)")
                ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, "per_category_gains.png"), dpi=150)
            plt.close(fig)
            top = G[G.condition == "D"].sort_values("purity_gain", ascending=False)
            md += [
                "Table: `results/per_category_gains.csv`; figure: `plots/per_category_gains.png`. Gains are means over the edited layers and over runs.",
                "",
                md_table(
                    G[G.condition == "D"].pivot(index="category", columns="arm", values="purity_gain").reset_index()
                ),
                "",
                f"Largest purity gain on unseen material: `{top.iloc[0].category}` ({top.iloc[0].purity_gain:+.4f}, {top.iloc[0].arm} arm); smallest: `{top.iloc[-1].category}` ({top.iloc[-1].purity_gain:+.4f}, {top.iloc[-1].arm} arm).",
                "",
            ]
        else:
            md += ["_No run with edited layers yet._", ""]

    def _no_runs(self) -> None:
        """Placeholder sections when no run has been saved yet."""
        md = self.md
        md += ["_No ablation runs found yet._", ""]
        self.base = pd.DataFrame()
        for study, (title, isolates, setup, by) in STUDY.items():
            md += [
                f"## {title}",
                "",
                f"**Purpose.** {isolates}",
                "",
                f"**Setup.** {setup}",
                "",
                self._not_run(study),
                "",
            ]
        md += [
            "## Per-layer trajectories",
            "",
            "_Produced for every default-configuration run that edits at least one layer._",
            "",
            "## Category-level breakdown",
            "",
            "_Produced from the same runs._",
            "",
        ]

    def _efficiency_and_transfer(self) -> None:
        """Analyst efficiency and transfer matrix: ablation runs with default settings, plus the main study when given."""
        md, base = self.md, self.base
        E = pd.DataFrame()
        if len(base):
            E = base[
                ["arm", "analyst", "target", "edited_layers", "cum_kl"] + [f"knn_purity_change_{c}" for c in "ABCD"]
            ].assign(source="ablation")
        if self.main_results and os.path.exists(self.main_results):
            m = pd.read_csv(self.main_results)
            m = pd.DataFrame(
                {
                    "arm": m["arm"],
                    "analyst": m["analyst"],
                    "target": m["target"],
                    "edited_layers": m["edited layers"],
                    "cum_kl": m["cumulative KL"],
                    "knn_purity_change_A": m["change knn_purity [A]"],
                    "knn_purity_change_D": m["change knn_purity [D]"],
                    "source": "main study",
                }
            )
            E = pd.concat([E, m], ignore_index=True)
        self.E = E
        md += ["## Analyst efficiency", ""]
        if not len(E):
            md += ["_No runs yet._", ""]
            return
        plt = _pyplot()
        fig, ax = plt.subplots(figsize=(8.5, 6))
        cm = plt.get_cmap("tab10")
        for i, (a, g) in enumerate(E.groupby("analyst")):
            ax.scatter(
                g.cum_kl,
                g.knn_purity_change_D,
                s=20 + 22 * g.edited_layers,
                alpha=0.6,
                color=cm(i % 10),
                label=a,
                edgecolor="black",
                linewidth=0.4,
            )
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("cumulative KL to the original model (behavioural cost)")
        ax.set_ylabel("kNN purity gain on unseen words in unseen frames (D)")
        ax.set_title("Analyst efficiency: one bubble per run, size = number of accepted edits")
        ax.legend(title="analyst", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "analyst_efficiency.png"), dpi=150)
        plt.close(fig)
        A = (
            E.groupby(["arm", "analyst"])
            .agg(
                runs=("target", "size"),
                accepted_edits=("edited_layers", "mean"),
                purity_gain_D=("knn_purity_change_D", "mean"),
                cum_kl=("cum_kl", "mean"),
            )
            .reset_index()
        )
        A["purity_gain_D_per_nat_KL"] = [g / k if k > 1e-6 else 0.0 for g, k in zip(A.purity_gain_D, A.cum_kl)]
        self._csv(A, "analyst_efficiency.csv")
        md += [
            "Figure: `plots/analyst_efficiency.png`; table: `results/analyst_efficiency.csv` (means per analyst; the last column is gain per nat of KL, 0 when the analyst changed nothing).",
            "",
            md_table(A),
            "",
        ]
        conds = [c for c in "ABCD" if f"knn_purity_change_{c}" in E and E[f"knn_purity_change_{c}"].notna().any()]
        TM = E.groupby(["arm", "analyst"])[[f"knn_purity_change_{c}" for c in conds]].mean().fillna(0.0)
        for c in conds[1:]:
            TM[f"transfer_{c}/A"] = [
                v / a if abs(a) >= 1e-3 else 0.0
                for v, a in zip(TM[f"knn_purity_change_{c}"], TM["knn_purity_change_A"])
            ]
        TM = TM.reset_index()
        self._csv(TM, "transfer_matrix.csv")
        Z = TM[[f"knn_purity_change_{c}" for c in conds]].values
        fig, ax = plt.subplots(figsize=(1.8 * len(conds) + 4, 0.55 * len(TM) + 2))
        v = max(abs(Z).max(), 1e-6)
        im = ax.imshow(Z, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(
            [
                {
                    "A": "A: probe words\n(in-sample)",
                    "B": "B: unseen words",
                    "C": "C: unseen frames",
                    "D": "D: unseen words\n+ frames",
                }[c]
                for c in conds
            ]
        )
        ax.set_yticks(range(len(TM)))
        ax.set_yticklabels([f"{r.arm} | {r.analyst}" for r in TM.itertuples()])
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, f"{Z[i, j]:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, label="change in kNN purity (edited - original)")
        ax.set_title("Transfer matrix: edits fitted on A, measured on each condition")
        fig.tight_layout()
        fig.savefig(os.path.join(self.plots_dir, "transfer_matrix.png"), dpi=150)
        plt.close(fig)
        md += [
            "## Transfer matrix",
            "",
            "Edits are chosen and fitted on condition A only. Each row is one analyst; the columns give the purity change on each condition and the ratio to A "
            "(0 when the in-sample change is too small to divide by)."
            + (" The main-study runs only recorded conditions A and D." if "main study" in set(E.source) else ""),
            "",
            "Table: `results/transfer_matrix.csv`; figure: `plots/transfer_matrix.png`.",
            "",
            md_table(TM),
            "",
        ]
