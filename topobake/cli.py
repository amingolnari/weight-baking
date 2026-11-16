"""
Topology-aware weight baking - full ablation suite (command-line interface).

The suite runs the complete layer-by-layer pipeline (model loading, descriptors, analyst prompt, verification, hook, closed-form weight
edit, safety limits) AND every ablation study, the evaluation and the analysis. Its only inputs are the four data files in --data-dir
(written by scripts/build_data.py):

    probe_words.txt            word <TAB> category, one per line (or words only + probe_labels.txt). The first 8 categories
                               in order of first appearance are the core set; 16 are needed for the category ablation.
    calibration_sentences.txt  one sentence per line; used to fit the weight edits
    heldout_sentences.txt      one sentence per line; the first 24 are the in-loop guard set, the rest is evaluation only
    new_domain_sentences.txt   one sentence per line; evaluation only (code, medical, legal, chat, ...)
Lines starting with '#' are comments.

Usage
    python scripts/run_ablation.py                                   # every study, 3 seeds, then tables, plots and the summary
    python scripts/run_ablation.py --ablation-type non_sequential    # one study (several names may be given)
    python scripts/run_ablation.py --analysts gemma-2-2b-it llama-3.2-3b-it --targets qwen2.5-1.5b-it smollm2-1.7b-it phi-3-mini-it
    python scripts/run_ablation.py --analyze-only                    # rebuild tables, plots and the summary from saved runs

Studies (--ablation-type; default: all)
    baseline / repeats   default configuration repeated over seeds: confidence intervals and p-values
    non_sequential       sequential loop vs a parallel control (all layers analysed on the original model, edits applied together)
    analyst_type         LLM-only vs rule-only vs hybrid (LLM first, rules when the LLM fails), as separate edited models
    descriptors          analyst input: purity+silhouette | +anisotropy | +intrinsic dimension+Betti numbers | full
    strength             fixed strength alpha (--alphas) vs the adaptive search
    gain_threshold       gain threshold (--gains); the relaxed-threshold stage is off so that the threshold binds
    categories           number of probe categories (--category-counts)

How it runs
    Every run is one configuration (arm, analyst, target, mode, descriptors, alpha, gain threshold, categories, seed). Its
    results are stored under <results-dir>/runs/<run id>/ and summarised in summary.json. A configuration is executed once,
    however many studies contain it; finished runs are skipped on restart; an interrupted sequential run resumes at the layer
    where it stopped. A seed changes the probe/unseen word split, the benchmark samples and the HellaSwag monitor sample.
    If a run edits no layer the model is unchanged: the before/after evaluation is skipped and every change is recorded as 0.

Outputs
    <results-dir>/   all_runs.csv, manifest.csv, repeated_runs_stats.csv, repeated_runs_by_pair.csv, ablation_non_sequential.csv,
                     ablation_analyst_type.csv, ablation_descriptors.csv, ablation_strength.csv, ablation_gain_threshold.csv,
                     ablation_categories.csv, per_category_gains.csv, transfer_matrix.csv, analyst_efficiency.csv
    <plots-dir>/     ablation_<study>.png, trajectory_<run id>.png, analyst_efficiency.png, per_category_gains.png, transfer_matrix.png
    ABLATION_SUMMARY.md   each study: purpose, setup, table of means with 95% intervals, findings read off the table

Locations (all relative to the working directory unless given): ./data, ./models, ./cache, ./results, ./plots and
<results-dir>/ABLATION_SUMMARY.md. The environment variable TOPOBAKE_HOME, when set, is used as the base folder of these defaults;
an explicit flag always wins. Pool models are cached under <models-dir>.
"""

import argparse
import os
import traceback
from dataclasses import replace
from typing import List, Optional

import pandas as pd

from .analysis import AblationReport
from .config import MODELS, Paths, PipelineConfig
from .geometry import make_ripser_fn
from .pipeline import ExperimentRunner
from .studies import DEFAULTS, STUDY_ARMS, plan
from .utils import configure_runtime, default_device, free_gpu


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface of the ablation suite."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--ablation-type",
        nargs="+",
        default=["all"],
        choices=list(STUDY_ARMS) + ["all"],
        help="studies to run (default: all)",
    )
    ap.add_argument("--data-dir", help="folder with probe_words.txt and the three sentence files (default: ./data)")
    ap.add_argument("--results-dir", help="results folder (default: ./results)")
    ap.add_argument("--plots-dir", help="figures folder (default: ./plots)")
    ap.add_argument("--summary", help="summary file (default: <results-dir>/ABLATION_SUMMARY.md)")
    ap.add_argument("--models-dir", help="folder where pool models are downloaded (default: ./models)")
    ap.add_argument("--cache-dir", help="folder of the benchmark dataset cache (default: ./cache)")
    ap.add_argument(
        "--analysts",
        nargs="+",
        default=["gemma-2-2b-it"],
        help="analyst models (names from MODELS) for the llm and hybrid arms",
    )
    ap.add_argument(
        "--targets", nargs="+", default=["qwen2.5-1.5b-it", "smollm2-1.7b-it"], help="target models (names from MODELS)"
    )
    ap.add_argument(
        "--arms", nargs="+", choices=["llm", "rule", "hybrid"], help="override the arms of the selected studies"
    )
    ap.add_argument("--num-runs", type=int, default=3, help="repetitions per configuration; run k uses seed seed0+k")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument(
        "--num-categories",
        type=int,
        default=8,
        help="categories used by every study except 'categories' (the first N of probe_words.txt)",
    )
    ap.add_argument("--category-counts", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5, 1.75, 2.0])
    ap.add_argument("--gains", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--holdout", type=float, default=0.2, help="share of each category's words kept as unseen words")
    ap.add_argument("--n-bench", type=int, help=f"examples per benchmark (default {PipelineConfig().n_bench})")
    ap.add_argument(
        "--main-results",
        help="runs_summary.csv of the main study, to add its runs to the analyst-efficiency plot and the transfer matrix",
    )
    ap.add_argument(
        "--analyze-only",
        action="store_true",
        help="do not run models; rebuild tables, plots and the summary from the saved runs",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    """Plan the selected studies, run every distinct configuration once, then write tables, plots and the summary."""
    a = build_parser().parse_args(argv)
    configure_runtime()
    paths = Paths.resolve(
        data_dir=a.data_dir,
        models_dir=a.models_dir,
        cache_dir=a.cache_dir,
        results_dir=a.results_dir,
        plots_dir=a.plots_dir,
        summary_path=a.summary,
    )
    paths.ensure()
    pcfg = replace(PipelineConfig(), n_bench=a.n_bench) if a.n_bench else PipelineConfig()
    device = default_device()
    ripser_fn = make_ripser_fn(pcfg.topo_n)
    print(
        f"device: {device} | ripser: {'yes' if ripser_fn else 'no (Betti-1 reported as 0, h1_computed = 0)'} | pool: {[m['name'] for m in MODELS]}"
    )

    if not a.analyze_only:
        studies = [s for s in STUDY_ARMS if s != "repeats"] if "all" in a.ablation_type else a.ablation_type
        P = pd.DataFrame([r for s in studies for r in plan(s, a)])
        uniq = P.drop_duplicates("run_id")
        print(
            f"{len(P)} study entries -> {len(uniq)} distinct runs (a configuration shared by several studies runs once)"
        )
        print(P.groupby(["study", "arm", "variant"]).size().rename("runs").to_string())
        mpath = os.path.join(paths.results_dir, "manifest.csv")
        if os.path.exists(mpath):
            P = (
                pd.concat([pd.read_csv(mpath, keep_default_na=False), P.astype(str)])
                .astype(str)
                .drop_duplicates(["study", "variant", "run_id"])
            )
        P.to_csv(mpath, index=False)
        order = uniq.sort_values(
            ["arm", "analyst", "n_categories", "seed", "target"]
        )  # one analyst load per block, data reloaded as rarely as possible
        runner = ExperimentRunner(paths, pcfg, holdout=a.holdout, device=device, ripser_fn=ripser_fn)
        failed = {}
        for i, r in enumerate(order.to_dict("records"), 1):
            try:
                print(f"\n[{i}/{len(order)}]", end=" ")
                runner.run_one({k: r[k] for k in DEFAULTS})
            except Exception as e:
                failed[r["run_id"]] = str(e)[:300]
                print(f"[run] {r['run_id']} FAILED: {failed[r['run_id']]}")
                traceback.print_exc(limit=3)
                free_gpu()
        runner.close()
        print(
            f"\nfinished: {len(order) - len(failed)} runs | failed: {failed or 'none'} | nan-control: {runner.nan.counts or 'no cell had to be filled'}"
        )
    AblationReport(paths.results_dir, paths.plots_dir, paths.summary_path, paths.data_dir, a.main_results).write()
