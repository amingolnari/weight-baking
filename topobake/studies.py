"""Study plans: the default run configuration, the arms and variants of every study, the run id.

Every run is one configuration (arm, analyst, target, mode, descriptors, alpha, gain threshold, categories, seed). A configuration
is executed once, however many studies contain it.
"""

import hashlib
import json
from typing import Any, Dict, List, Mapping, Tuple, Union

from .config import RunConfig

DEFAULTS = RunConfig().to_dict()

STUDY_ARMS = {
    "baseline": ["llm", "rule"],
    "repeats": ["llm", "rule"],
    "non_sequential": ["llm", "rule"],
    "analyst_type": ["llm", "rule", "hybrid"],
    "descriptors": ["llm"],
    "strength": ["rule"],
    "gain_threshold": ["rule"],
    "categories": ["llm", "rule"],
}


def run_id(cfg: Union[RunConfig, Mapping[str, Any]]) -> str:
    """Stable identifier of a configuration: readable fields plus the first 8 hex digits of the MD5 of its JSON."""
    c = {**DEFAULTS, **(cfg.to_dict() if isinstance(cfg, RunConfig) else dict(cfg))}
    h = hashlib.md5(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{c['arm']}_{c['analyst']}_{c['target']}_{c['mode'][:3]}_{c['descriptors']}_a{c['alpha']}_g{c['min_gain']}{'' if c['relaxed'] else 'nr'}_c{c['n_categories']}_s{c['seed']}_{h}"


def variants(study: str, a: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """(label, config overrides) for every variant of a study. The first entry is the reference the others are compared with."""
    if study in ("baseline", "repeats"):
        return [("default", {})]
    if study == "non_sequential":
        return [("sequential", {"mode": "sequential"}), ("parallel", {"mode": "parallel"})]
    if study == "analyst_type":
        return [("default", {})]  # the arms are the variants
    if study == "descriptors":
        return [
            ("full", {"descriptors": "full"}),
            ("purity+silhouette", {"descriptors": "ps"}),
            ("+anisotropy", {"descriptors": "psa"}),
            ("+intrinsic dim+Betti", {"descriptors": "psa_topo"}),
        ]
    if study == "strength":
        return [("adaptive", {"alpha": "adaptive"})] + [(f"alpha={x:g}", {"alpha": x}) for x in a.alphas]
    if study == "gain_threshold":
        return [(f"gain>={x:g}", {"min_gain": x, "relaxed": False}) for x in a.gains]
    if study == "categories":
        return [(f"{n} categories", {"n_categories": n}) for n in a.category_counts]
    raise ValueError(study)


def plan(study: str, a: Any) -> List[Dict[str, Any]]:
    """One row (study, variant, run id and the full configuration) for every run of a study."""
    rows = []
    for arm in a.arms or STUDY_ARMS[study]:
        pairs = (
            [("rule-only", t) for t in a.targets]
            if arm == "rule"
            else [(x, t) for x in a.analysts for t in a.targets if x != t]
        )
        for label, over in variants(study, a):
            for analyst, target in pairs:
                for seed in range(a.seed0, a.seed0 + a.num_runs):
                    cfg = {
                        **DEFAULTS,
                        "n_categories": a.num_categories,
                        "arm": arm,
                        "analyst": analyst,
                        "target": target,
                        "seed": seed,
                        **over,
                    }
                    rows.append({"study": study, "variant": label, "run_id": run_id(cfg), **cfg})
    return rows
