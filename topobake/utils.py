"""Small helpers: model download, GPU cleanup, finite-number guards, NaN control and tensor utilities."""

import gc
import os
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch


def configure_runtime() -> None:
    """Silence library warnings for a command-line run (called by the CLI, never at import time)."""
    warnings.filterwarnings("ignore")


def default_device() -> str:
    """ "cuda" when a GPU is available, otherwise "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure(repo: str, base: str) -> str:
    """Download a repo to base unless a finished copy is already there; a marker file records a finished download."""
    from huggingface_hub import snapshot_download

    d = os.path.join(base, repo.split("/")[-1])
    mark = os.path.join(d, ".download_complete")
    if os.path.exists(mark):
        print(f"[cache] {repo}")
        return d
    print(f"[download] {repo} -> {d}")
    snapshot_download(
        repo, local_dir=d, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "tokenizer*", "*.jinja"]
    )
    open(mark, "w").close()
    return d


def free_gpu() -> None:
    """Run the garbage collector and release cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---- NaN control -----------------------------------------------------------------------------------
# (a) NaN is avoided at the source: targets run in float32, estimators are guarded against zero distances,
#     every log row carries every column, undefined ratios are reported as 0 with an explicit flag.
# (b) Every table passes through NanReport.clean() before it is saved or printed. clean() never hides anything:
#     the number of cells it had to fill is collected in NanReport.counts and printed at the end of the run.
class NanReport:
    """Collects, per table name, how many non-finite cells ``clean`` had to fill."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}

    def clean(self, df: pd.DataFrame, name: str, fill: float = 0.0) -> pd.DataFrame:
        """Replace inf/NaN in numeric columns by ``fill`` (and NaN elsewhere by "-"), recording the number of numeric cells filled."""
        df = df.copy()
        num = list(df.select_dtypes(include=[np.number]).columns)
        if num:
            df[num] = df[num].replace([np.inf, -np.inf], np.nan)
            n = int(df[num].isna().sum().sum())
            if n:
                self.counts[name] = self.counts.get(name, 0) + n
                df[num] = df[num].fillna(fill)
        other = [c for c in df.columns if c not in num]
        if other:
            df[other] = df[other].fillna("-")
        return df


def fin(x: Any, default: float = 0.0) -> float:
    """A finite float, or the default."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


def ratio(num: float, den: float, floor: float = 1e-3) -> float:
    """num/den, or 0 when the denominator is too small for the ratio to mean anything."""
    return fin(num / den) if abs(den) >= floor else 0.0


def short(tid: str) -> str:
    """Last path component of a repository id."""
    return tid.split("/")[-1]


def first(o: Any) -> Any:
    """The first element of a tuple output of a transformer block, or the output itself."""
    return o[0] if isinstance(o, tuple) else o


def kl(P: List[Any], Q: List[Any]) -> float:
    """KL(P || Q) in nats per token, averaged over lists of log-probability tensors."""
    return float(np.mean([(p.exp() * (p - q)).sum(-1).mean().item() for p, q in zip(P, Q)]))


def cos_shift(X: np.ndarray, Y: np.ndarray) -> float:
    """One minus the mean cosine similarity between matching rows of X and Y."""
    return float(1 - ((X * Y).sum(1) / np.linalg.norm(X, axis=1) / np.linalg.norm(Y, axis=1)).mean())


def pref(g: Dict[str, Any], p: str) -> Dict[str, Any]:
    """Prefix every key of g with ``p_``."""
    return {f"{p}_{k}": v for k, v in g.items()}
