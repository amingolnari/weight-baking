"""Configuration: storage paths, pipeline constants, run configuration and the model pool.

Nothing in this module touches the file system or the GPU at import time. Folders are created only when
``Paths.ensure()`` is called (by the command-line interface).
"""

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

# ---- model pool: round-robin, one analyst per round ----
# Every model is edited (target) by every other model (analyst). Optional keys per entry:
#   "target": False        analyst only            "analyst": False        target only
#   "analyst_repo": "..."  other weights for the analyst role (a "...-bnb-4bit" repo is loaded in 4-bit)
# Instruction-tuned models are used because each one also has to act as analyst.
# The entries are read-only: callers copy an entry with dict(entry) before adding keys.
MODELS = (
    {"name": "qwen2.5-1.5b-it", "repo": "Qwen/Qwen2.5-1.5B-Instruct"},  # 28 layers, d=1536
    {"name": "llama-3.2-3b-it", "repo": "unsloth/Llama-3.2-3B-Instruct"},  # 28 layers, d=3072 (ungated copy)
    {"name": "phi-3-mini-it", "repo": "microsoft/Phi-3-mini-4k-instruct"},  # 32 layers, d=3072
    {"name": "gemma-2-2b-it", "repo": "unsloth/gemma-2-2b-it"},  # 26 layers, d=2304 (ungated copy)
    {"name": "smollm2-1.7b-it", "repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct"},  # 24 layers, d=2048
)
# used only to replace a pool model that cannot be downloaded
SPARE_MODELS = (
    {"name": "qwen2.5-3b-it", "repo": "Qwen/Qwen2.5-3B-Instruct"},
    {"name": "llama-3.2-1b-it", "repo": "unsloth/Llama-3.2-1B-Instruct"},
)
MIN_MODELS = 5  # the study warns loudly if fewer models than this are available
# The rule-based proposer of the control round: it needs no model, so its pool entry has no weights.
RULE = {"name": "rule-only", "repo": "heuristic", "target": False, "arm": "rule"}

# A, B, C, D: A seen words/seen frame, B unseen/seen, C seen/unseen frames, D unseen/unseen frames
CONDITIONS = ("A", "B", "C", "D")

ENV_HOME = "TOPOBAKE_HOME"  # optional base folder for every default location below


@dataclass(frozen=True)
class Paths:
    """Storage locations. Defaults are relative to the current working directory.

    Precedence of ``resolve``: explicit argument (command-line flag), then the environment variable
    ``TOPOBAKE_HOME`` used as base folder, then the plain defaults ``data``, ``models``, ``cache``, ``results``, ``plots``.
    """

    data_dir: str = "data"  # the four text files written by scripts/build_data.py
    models_dir: str = "models"  # pool models (used both as target and as analyst)
    cache_dir: str = "cache"  # benchmark datasets cache
    results_dir: str = "results"  # runs/, CSV tables
    plots_dir: str = "plots"  # figures
    summary_path: str = os.path.join("results", "ABLATION_SUMMARY.md")

    @classmethod
    def resolve(
        cls,
        data_dir: Optional[str] = None,
        models_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        plots_dir: Optional[str] = None,
        summary_path: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> "Paths":
        """Combine command-line values (None = not given) with the optional base folder and the defaults."""
        env = os.environ if env is None else env
        home = env.get(ENV_HOME) or ""

        def pick(flag: Optional[str], name: str) -> str:
            return flag or (os.path.join(home, name) if home else name)

        results = pick(results_dir, "results")
        return cls(
            data_dir=pick(data_dir, "data"),
            models_dir=pick(models_dir, "models"),
            cache_dir=pick(cache_dir, "cache"),
            results_dir=results,
            plots_dir=pick(plots_dir, "plots"),
            summary_path=summary_path or os.path.join(results, "ABLATION_SUMMARY.md"),
        )

    @property
    def analyst_dir(self) -> str:
        """Weights used only for a separate "analyst_repo" (for example a 4-bit copy)."""
        return os.path.join(self.models_dir, "analysts")

    @property
    def edited_dir(self) -> str:
        """Where full edited checkpoints would be written (off by default)."""
        return os.path.join(self.models_dir, "edited")

    @property
    def runs_dir(self) -> str:
        """One sub-folder per run id."""
        return os.path.join(self.results_dir, "runs")

    def ensure(self) -> None:
        """Create the output folders (the data folder belongs to the user and is not created)."""
        for d in (self.models_dir, self.analyst_dir, self.cache_dir, self.runs_dir, self.plots_dir):
            os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.dirname(self.summary_path) or ".", exist_ok=True)


@dataclass(frozen=True)
class PipelineConfig:
    """Every numeric limit and constant of the pipeline. Per-run variations are made with ``dataclasses.replace``."""

    analyst_dtype: str = "bfloat16"  # analysts are only read, never edited
    target_dtype: str = (
        "float32"  # targets are edited: full precision, so small weight changes are not rounded away and no overflow -> no NaN
    )

    # ---- probe ----
    k: int = 8
    template: str = "I thought about the {w}"
    cloze: str = 'The word "{w}" refers to a kind of'

    # ---- analyst loop, edit budget, skipping, safety ----
    max_attempts: int = 3
    max_new_tokens: int = 400
    scales: Tuple[float, ...] = (1.0, 0.5, 0.25)
    max_ops: int = 4
    max_rel_change: float = 0.20
    min_gain: float = 0.01
    max_rel_weight_change: float = 0.10
    ridges: Tuple[float, ...] = (1e-2, 1e-1, 1.0)
    skip_purity: float = 0.95
    skip_cross: float = 0.05
    max_step_kl: float = 0.05
    max_cum_kl: float = 0.30

    # ---- adaptive strength ----
    adapt_grid: Tuple[float, ...] = (
        1.5,
        1.25,
        1.0,
        0.75,
        0.5,
        0.25,
    )  # every proposal is evaluated at these strengths; the best safe one is kept
    max_hook_tests: int = 4  # model-level tests per proposal (each is one snapshot)

    # ---- perplexity preservation ----
    max_step_ppl_increase: float = 0.02  # one edit may raise held-out perplexity by at most 2 %
    max_cum_ppl_increase: float = 0.05  # the edited model may be at most 5 % above the original model

    # ---- representation "essence" ----
    min_step_cka: float = 0.95  # linear CKA of a layer before vs after its own edit (probe words and held-out tokens)
    n_monitor: int = (
        100  # HellaSwag examples checked after every baked edit (disjoint from the final evaluation subset)
    )
    max_hellaswag_drop: float = (
        0.05  # the edited model may lose at most this much monitor accuracy against the original model
    )

    # ---- evaluation ----
    n_bench: Optional[int] = 200  # examples per benchmark (HellaSwag, ARC-Easy, TruthfulQA); None = full split
    benchmarks: Tuple[str, ...] = ("hellaswag", "arc_easy", "truthfulqa")

    # ---- run control ----
    resume: bool = True  # continue interrupted targets; False = redo every target from its original weights
    redo_evaluation: bool = False  # recompute the evaluation of targets that are already finished
    max_layers_per_target: Optional[int] = None  # None = every layer
    save_checkpoint: bool = False  # write the full edited model at the end of a run (weight deltas are always saved)

    # ---- data sizes ----
    guard_n: int = 24  # in-loop guard sentences
    keep_lp: int = 24  # texts whose log-probs are kept in memory
    keep_hs: int = 12  # texts whose hidden states are kept in memory
    topo_n: int = 300  # points used for persistent homology and intrinsic dimension (stratified subsample)
    calib_words: int = 200  # probe-word prompts added to the calibration sentences for the weight edit
    max_tokens: int = 96  # sentences are truncated to this many tokens


@dataclass(frozen=True)
class RunConfig:
    """One run: the keys of the former ``cfg`` dict (their defaults are the former ``DEFAULTS``).

    arm           "llm" | "rule" | "hybrid"           which proposer(s) may write edits
    analyst       pool name of the analyst model, or "rule-only"
    target        pool name of the edited model
    mode          "sequential" | "parallel"
    descriptors   "ps" | "psa" | "psa_topo" | "full"  what the analyst is shown
    alpha         "adaptive" or a number              strength: the adaptive grid, or one fixed strength
    min_gain      float                               gain threshold
    relaxed       bool                                allow the relaxed-threshold stage
    n_categories  int                                 how many categories of probe_words.txt are used (the first N)
    seed          int                                 probe/unseen split, benchmark sample, HellaSwag monitor sample
    """

    arm: str = "llm"
    analyst: str = "gemma-2-2b-it"
    target: str = "qwen2.5-1.5b-it"
    mode: str = "sequential"
    descriptors: str = "full"
    alpha: Union[str, float] = "adaptive"
    min_gain: float = 0.01
    relaxed: bool = True
    n_categories: int = 8
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """The configuration as a plain dict in the key order of the defaults."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any]) -> "RunConfig":
        """Fill the defaults into a (possibly partial) mapping."""
        return cls(**{**asdict(cls()), **dict(m)})
