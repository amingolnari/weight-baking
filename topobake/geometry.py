"""Geometry of point clouds: cosine descriptors, CKA, intrinsic dimension, persistent homology and the analyst digest.

``ripser`` is optional: without it Betti-1 is reported as 0 and ``h1_computed`` is 0 (Betti-0 is exact and always available).
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples

from .config import PipelineConfig
from .data import ProbeData
from .utils import fin

try:
    from ripser import ripser as _ripser
except ImportError:
    _ripser = None


def make_ripser_fn(topo_n: int) -> Optional[Callable[..., Any]]:
    """Persistent homology on at most ``topo_n`` points (H1 on thousands of points is too slow); None when ripser is missing."""
    if _ripser is None:
        return None

    def ripser_sub(Dc, **kw):
        if len(Dc) > topo_n:
            i = np.linspace(0, len(Dc) - 1, topo_n).astype(int)
            Dc = Dc[np.ix_(i, i)]
        return _ripser(Dc, **kw)

    return ripser_sub


# ====================================================================================================
# Extended metrics
# ====================================================================================================
# **Linear CKA** (`cka`). Centered Kernel Alignment between the same inputs represented by the original and by the edited model at one layer. 1 means the two representations are identical up to rotation and scale; lower values mean the edit changed *which* inputs are similar to which, not just the coordinate system. It is the complement of the purity gain: a good edit raises purity while keeping CKA high.
#
# **Intrinsic dimension** (`topo`). Two estimators that fail in different ways: *Two-NN* (ratio of the two nearest-neighbour distances; local, sensitive to noise) and *correlation dimension* (slope of the log correlation integral over small radii; more global). Agreement between them is more informative than either alone.
#
# **Persistent homology** (`topo`). From the cosine-distance filtration: Betti-0 (number of connected components) and Betti-1 (number of loops) read at two scales - the 10th and 25th percentile of all pairwise distances - plus the total and maximum persistence of loops. With 8 categories, a Betti-0 near 8 at a moderate scale is what clean category structure looks like; a high Betti-1 indicates ring-like tangles. Betti-0 is exact (from the minimum spanning tree); Betti-1 needs `ripser` and is empty without it.


def cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same n inputs (rows)."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    Kx, Ky = X @ X.T, Y @ Y.T
    den = np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum())
    return fin((Kx * Ky).sum() / den, 1.0) if den > 0 else 1.0


def topo(X: np.ndarray, ripser_fn: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Intrinsic dimension and persistent-homology summaries of a point cloud."""
    n = len(X)
    r = np.maximum(pdist(X), 1e-12)
    es = np.sort(squareform(r), 1)[:, 1:3]
    ps = np.linspace(0.02, 0.30, 8)
    corr = np.polyfit(np.log(np.quantile(r, ps)), np.log(ps), 1)[0]  # C(r) at the p-quantile of distances is p
    dc = pdist(X, "cosine")
    Dc = squareform(dc)
    q10, q25 = np.quantile(dc, [0.10, 0.25])
    h0 = minimum_spanning_tree(Dc).data  # H0 death scales
    out = {
        "twonn_id": fin(n / max(np.log(es[:, 1] / es[:, 0]).sum(), 1e-12)),
        "corr_dim": fin(corr),
        "betti0_q10": n - int((h0 <= q10).sum()),
        "betti0_q25": n - int((h0 <= q25).sum()),
        "betti1_q10": 0,
        "betti1_q25": 0,
        "h1_total_persistence": 0.0,
        "h1_max_persistence": 0.0,
        "h1_computed": int(bool(ripser_fn)),
    }
    if ripser_fn:
        h1 = ripser_fn(Dc, distance_matrix=True, maxdim=1)["dgms"][1]
        pers = h1[:, 1] - h1[:, 0] if len(h1) else np.zeros(0)
        out.update(
            betti1_q10=int(((h1[:, 0] <= q10) & (h1[:, 1] > q10)).sum()),
            betti1_q25=int(((h1[:, 0] <= q25) & (h1[:, 1] > q25)).sum()),
            h1_total_persistence=float(pers.sum()),
            h1_max_persistence=float(pers.max()) if len(pers) else 0.0,
        )
    return out


# ====================================================================================================
# Fast geometry and lean descriptors
# ====================================================================================================
def geom_parts(X: np.ndarray, cats: np.ndarray, K: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine geometry from one Gram matrix: anisotropy, per-point kNN purity, per-point silhouette, neighbour indices, distance matrix."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    D = np.clip(1 - Xn @ Xn.T, 0, 2)
    np.fill_diagonal(D, 0)
    n = len(X)
    nb = np.argsort(D + 9 * np.eye(n), 1)[:, :K]
    pur = (cats[nb] == cats[:, None]).mean(1)
    sil = silhouette_samples(D, cats, metric="precomputed") if len(set(cats)) > 1 else np.zeros(n)
    return 1 - D[np.triu_indices(n, 1)].mean(), pur, sil, nb, D


def geom(X: np.ndarray, cats: np.ndarray, K: int = 8) -> Dict[str, float]:
    """Headline geometry of a point cloud: anisotropy, kNN purity, silhouette."""
    a, pur, sil, _, _ = geom_parts(X, cats, K)
    return {"anisotropy": float(a), "knn_purity": float(pur.mean()), "silhouette": float(sil.mean())}


def rel_change(X: np.ndarray, Y: np.ndarray) -> float:
    """Mean relative displacement, the edit-budget measure."""
    return float((np.linalg.norm(Y - X, axis=1) / np.linalg.norm(X, axis=1)).mean())


def describe(
    X: np.ndarray, L: int, data: ProbeData, pcfg: PipelineConfig, ripser_fn: Optional[Callable[..., Any]] = None
) -> Dict[str, pd.DataFrame]:
    """Lean descriptor tables for large probe sets: exactly what the analyst digest and the rule-based proposer read."""
    cats, words, K, C = data.cats, data.words, pcfg.k, data.categories
    aniso, pur, sil, nb, D = geom_parts(X, cats, K)
    n = len(X)
    sub = data.sub
    t = topo(X[sub], ripser_fn)
    Sv = np.linalg.svd(X[sub] - X[sub].mean(0), compute_uv=False)
    var = Sv**2 / max((Sv**2).sum(), 1e-12)
    cen = np.stack([X[cats == c].mean(0) for c in C])
    cn = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    Cd = 1 - cn @ cn.T
    share = (cats[nb][:, :, None] == np.array(C)[None, None]).mean(1)
    cross = np.stack([share[cats == c].mean(0) for c in C])
    pairs = pd.DataFrame(
        [
            {
                "layer": L,
                "a": C[p],
                "b": C[q],
                "centroid_cos_dist": Cd[p, q],
                "cross_knn_rate": (cross[p, q] + cross[q, p]) / 2,
            }
            for p in range(len(C))
            for q in range(p + 1, len(C))
        ]
    )
    pts = pd.DataFrame(
        {
            "layer": L,
            "word": words,
            "category": cats,
            "norm": np.linalg.norm(X, axis=1),
            "knn_purity": pur,
            "silhouette": sil,
            "nearest": words[nb[:, 0]],
        }
    )
    same = cats[:, None] == cats[None]
    off = ~np.eye(n, dtype=bool)
    summ = {
        "layer": L,
        "anisotropy": float(aniso),
        "knn_purity": float(pur.mean()),
        "silhouette": float(sil.mean()),
        "mean_norm": float(pts.norm.mean()),
        "top_pc_var": float(var[0]),
        "effective_dim": float((Sv**2).sum() ** 2 / max((Sv**4).sum(), 1e-12)),
        "twonn_intrinsic_dim": t["twonn_id"],
        "corr_dim": t["corr_dim"],
        "intra_inter_ratio": float(D[same & off].mean() / max(D[~same].mean(), 1e-12)),
        "betti0_q25": t["betti0_q25"],
        "betti1_q25": t["betti1_q25"],
        "h1_computed": t["h1_computed"],
        "h1_loops": t["betti1_q25"],
        "h1_max_persistence": t["h1_max_persistence"],
    }
    return {
        "layer_summary": pd.DataFrame([summ]),
        "points": pts,
        "category_pairs": pairs,
        "pca_spectrum": pd.DataFrame({"layer": L, "pc": np.arange(len(var)), "explained_var": var}),
    }


DESCRIPTOR_LEVELS = {"ps": 0, "psa": 1, "psa_topo": 2, "full": 3}


def digest(
    L: int,
    d: Dict[str, pd.DataFrame],
    pre: Dict[str, Any],
    *,
    descriptors: str,
    n_layers: int,
    cum_kl: float,
    base_ppl: float,
    applied: List[Tuple[int, Dict[str, Any]]],
    pcfg: PipelineConfig,
) -> str:
    """Text shown to the analyst. ``descriptors`` decides which descriptor families it contains:
       ps        kNN purity + silhouette (global, per category, conflated pairs, least pure words)
       psa       + anisotropy family (anisotropy, norm, PCA spectrum)
       psa_topo  + intrinsic dimension and Betti numbers
       full      + drift caused by upstream edits and the intra/inter distance ratio
    Edit history and the behaviour budgets are procedural, not descriptors, and are always shown.
    ``applied`` lists the (layer, log row) pairs of the edits applied so far; ``cum_kl`` is the cumulative KL to the original model.
    """
    lv = DESCRIPTOR_LEVELS[descriptors]
    s = d["layer_summary"].iloc[0]
    p = d["category_pairs"]
    pts = d["points"]
    top = p.sort_values("cross_knn_rate", ascending=False).head(6)
    low = pts.nsmallest(8, "knn_purity")
    cum = cum_kl
    hist = [
        f"  layer {l}: {r['spec']} | knn_purity {r['before_knn_purity']:.3f}->{r['after_knn_purity']:.3f} | step KL {r['step_kl']:.4f}"
        for l, r in applied
    ][-5:]
    out = [f"LAYER {L} of {n_layers}", "metric: current (original model)"]
    out += [
        f"  {m}: {s[m]:.3f} ({s['orig_' + m]:.3f})"
        for m in ["knn_purity", "silhouette"] + (["anisotropy"] if lv >= 1 else [])
    ]
    if lv >= 1:
        out += [
            f"  mean_norm: {s.mean_norm:.1f}",
            "explained variance of PC0..PC5: "
            + ", ".join(f"{v:.3f}" for v in d["pca_spectrum"].explained_var.values[:6]),
        ]
    if lv >= 2:
        out += [
            f"  effective_dim: {s.effective_dim:.1f} | intrinsic_dim (Two-NN): {s.twonn_intrinsic_dim:.1f} | correlation dim: {s.corr_dim:.1f}",
            f"  Betti numbers at the 25th-percentile distance: b0 = {int(s.betti0_q25)}, b1 = {int(s.betti1_q25) if s.h1_computed else 'not computed'}",
        ]
    if lv >= 3:
        out += [
            f"  intra_inter_ratio: {s.intra_inter_ratio:.3f}",
            f"  drift from original caused by upstream edits: rel_l2 {s.drift_rel_l2:.3f}, cos {s.drift_cos:.4f}",
        ]
    out += [
        f"most conflated category pairs (median centroid_cos_dist over all pairs = {p.centroid_cos_dist.median():.3f}):",
        *[
            f"  {r.a} - {r.b}: cross_knn_rate {r.cross_knn_rate:.3f}, centroid_cos_dist {r.centroid_cos_dist:.3f}"
            for r in top.itertuples()
        ],
        "knn_purity per category: "
        + ", ".join(f"{c} {v:.2f}" for c, v in pts.groupby("category").knn_purity.mean().items()),
        "least pure words (word [category] purity -> nearest neighbour): "
        + "; ".join(f"{r.word} [{r.category}] {r.knn_purity:.2f} -> {r.nearest}" for r in low.itertuples()),
        "recent upstream edits:",
        *(hist or ["  none"]),
        f"behaviour budget: cumulative KL so far {cum:.4f} of {pcfg.max_cum_kl}; this edit may cost at most {max(min(pcfg.max_step_kl, pcfg.max_cum_kl - cum), 0):.4f}",
        f"perplexity budget: {pre['ppl']:.2f} now, {base_ppl:.2f} originally ({pre['ppl'] / base_ppl - 1:+.1%}); limit {pcfg.max_cum_ppl_increase:+.0%} in total, "
        f"{pcfg.max_step_ppl_increase:+.0%} for this edit",
        f"Return the JSON object for layer {L}.",
    ]
    return "\n".join(out)
