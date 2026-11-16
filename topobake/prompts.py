"""The analyst's texts: the brief (SYSTEM), the forced JSON start (PREFIX), the reference implementation shown as text.

The brief is a template with empty category and ambiguous-word lists; ``render_system`` rewrites it for the probe set and the gain
threshold of one run (number of probe words, category names, gain threshold) exactly as the original suite did.
"""

import re
from dataclasses import replace

from .config import PipelineConfig

AMBIGUOUS = (
    frozenset()
)  # words whose neighbours in another category are correct; the brief names them generically at run time

# ====================================================================================================
# Transform operations - reference implementation
# ====================================================================================================
# One definition of what a spec means, kept as text. The executable versions are edits.build_affine / edits.apply_spec and
# geometry.rel_change (the lean versions that also cache the SVD).

REF_IMPL = r"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

def geom(X, cats, K=8):
    # headline geometry of a point cloud X (n, d) with labels cats (n,)
    Dc = squareform(pdist(X, "cosine")); nb = np.argsort(Dc, 1)[:, 1:K + 1]
    return {"anisotropy": float(1 - pdist(X, "cosine").mean()),
            "knn_purity": float((cats[nb] == cats[:, None]).mean()),
            "silhouette": float(silhouette_score(X, cats, metric="cosine"))}

def build_affine(spec, X, cats):
    # compile spec["ops"] into (A, b) such that h' = h @ A + b ; all statistics come from X (the layer BEFORE the edit)
    d = X.shape[1]; mu = X.mean(0); S, Vt = np.linalg.svd(X - mu, full_matrices=False)[1:]
    A, b = np.eye(d), np.zeros(d)
    def push(U, f, c):                      # scale by f along the orthonormal rows of U, about the point c
        nonlocal A, b
        M = np.eye(d) + (U.T * (np.asarray(f, float) - 1)) @ U
        A, b = A @ M, b @ M + c - c @ M
    for o in spec["ops"]:
        if o["op"] == "scale_pcs": push(Vt[o["pcs"]], o["factor"], mu)
        elif o["op"] == "whiten":
            s = S[:o["rank"]]; push(Vt[:o["rank"]], (np.exp(np.log(s).mean()) / s) ** o.get("strength", 1), mu)
        elif o["op"] == "separate":
            ca, cb = X[cats == o["a"]].mean(0), X[cats == o["b"]].mean(0)
            push(((ca - cb) / np.linalg.norm(ca - cb))[None], o["factor"], (ca + cb) / 2)
        elif o["op"] == "center": b = b - o.get("strength", 1) * (mu @ A + b)
        else: raise ValueError("unknown op " + str(o["op"]))
    return A, b

def apply_spec(spec, X, cats):
    A, b = build_affine(spec, X, cats); Y = X @ A + b
    if spec.get("preserve_norm"): Y = Y * np.linalg.norm(X, axis=1, keepdims=True) / np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def rel_change(X, Y):                       # mean relative displacement, the edit-budget measure
    return float((np.linalg.norm(Y - X, axis=1) / np.linalg.norm(X, axis=1)).mean())
"""

# ====================================================================================================
# The analyst: prompt
# ====================================================================================================
# The geometric-architect brief. The brief states the perplexity and CKA limits and tells the analyst that the system tunes the
# overall strength itself.


def system_template(pcfg: PipelineConfig, categories=(), ambiguous=AMBIGUOUS) -> str:
    """The analyst brief for the constants of ``pcfg``; the category and ambiguous-word lists are empty until rendered."""
    return f"""You are a geometric architect for the latent space of a frozen language model.
You receive numerical descriptors of ONE layer (the residual stream after one transformer block) for 80 probe
words in 8 categories ({", ".join(categories)}) and must prescribe a small edit of that layer's feature space.

CONTEXT
- Layers are edited one at a time from first to last. Earlier edits are already baked into the weights and the
  descriptors you see describe the CURRENT, already-edited model.
- Your edit is applied to every token at this layer and then compiled into the weights. Every later layer will
  receive the changed features, so prefer the smallest edit that fixes a clear distortion.
- You cannot run code. The system simulates your proposal and tests it on the model; if it fails you receive the
  measured numbers and must return a corrected proposal. Never include predicted numbers.
- The system tunes the overall strength of your proposal itself (0.25x to 1.5x of your factors) and keeps the
  strongest version that passes every limit. Concentrate on WHICH operations and category pairs to use and on
  their relative sizes.

DEFINITIONS
anisotropy = mean pairwise cosine similarity (near 1 = all vectors almost parallel).
knn_purity = share of the {pcfg.k} nearest neighbours that have the same category (higher is better).
cross_knn_rate(a,b) = share of neighbour links from one category that land in the other (high = conflated).
centroid_cos_dist(a,b) = cosine distance between category centroids (small = conflated).

ALLOWED OPERATIONS (applied in order; statistics come from this layer before the edit)
{{"op":"separate","a":"<category>","b":"<category>","factor":f}}  stretch (f>1) or contract (f<1) along the line joining
    the two category centroids. f in [0.5, 2.0]. Use for conflated category pairs.
{{"op":"scale_pcs","pcs":[i,...],"factor":f}}  multiply the listed 0-based principal components by f. f in [0, 2].
    Use only when one PC clearly dominates the variance.
{{"op":"whiten","rank":r,"strength":s}}  equalise the variance of the top r PCs. r in [1, 16], s in [0, 0.5].
{{"op":"center","strength":s}}  subtract s times the mean vector. s in [0, 0.5]. Changes every token strongly.

HARD LIMITS
- At most {pcfg.max_ops} operations. Category names exactly as listed above.
- The mean relative change of the features must stay below {pcfg.max_rel_change}; factors near 1 are safer.
- The edit must raise knn_purity + silhouette by at least {pcfg.min_gain}, otherwise it is discarded.
- Held-out perplexity may rise by at most {pcfg.max_step_ppl_increase:.0%} for this edit and {pcfg.max_cum_ppl_increase:.0%} in total against the original model.
- The layer must stay similar to itself: linear CKA before/after the edit at least {pcfg.min_step_cka}.
- Ambiguous words ({", ".join(sorted(ambiguous))}) having neighbours in another category is correct, not a distortion.
- If the layer needs no edit, return "ops": [].

OUTPUT: one JSON object and nothing else - no prose, no markdown, no code fences:
{{"layer": <int>, "preserve_norm": true, "ops": [...], "rationale": "<max 2 sentences citing the numbers used>"}}"""


def render_system(pcfg: PipelineConfig, n_words: int, categories) -> str:
    """The analyst brief adapted to the current probe set (word and category counts, names) and gain threshold."""
    s = system_template(
        replace(pcfg, min_gain=PipelineConfig.min_gain)
    )  # the pristine brief: neutral lists, default gain
    s = re.sub(
        r"for \d+ probe\s+words in \d+ categories \([^)]*\)",
        f"for {n_words} probe\nwords in {len(categories)} categories ({', '.join(categories)})",
        s,
    )
    s = re.sub(r"- Ambiguous words \([^)]*\)", "- Ambiguous words (words with several senses)", s)
    return re.sub(r"by at least [\d.]+, otherwise", f"by at least {pcfg.min_gain}, otherwise", s)


def prefix(layer: int) -> str:
    """The reply is forced to start as the JSON object."""
    return f'{{"layer": {layer}, "preserve_norm": true, "ops": ['
