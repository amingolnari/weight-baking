"""Probe set, calibration text, unseen material, the factual-preference probe and the evaluation benchmarks.

Probe words and sentences are NOT hard-coded: ``ProbeData.load`` reads them from the four text files written by
``scripts/build_data.py``. Only the sentence frames, the cloze question and the factual probe are defined here.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .config import PipelineConfig

GEN_TEMPLATES = [
    "The technical report included a short section about the {w}",
    "During the court hearing, the witness mentioned the {w}",
    "The next slide of the lecture shows a picture of the {w}",
]
GEN_CLOZE = 'Question: what kind of thing is "{w}"? Answer: it is a kind of'

# ====================================================================================================
# Factual-preference probe
# ====================================================================================================
# Forty short factual prompts, each with the true completion and one plausible false completion. For every prompt the model's log-probability of both completions is compared. This is a *proxy* for hallucination: it measures how often the model prefers a false statement to the true one, and how confidently, without relying on free-text generation and a judge.

FACTS = [l.split("|") for l in """The capital of France is| Paris| Rome
The capital of Japan is| Tokyo| Beijing
The capital of Italy is| Rome| Madrid
The capital of Germany is| Berlin| Vienna
The capital of Egypt is| Cairo| Baghdad
The capital of Canada is| Ottawa| Toronto
The capital of Australia is| Canberra| Sydney
The capital of Russia is| Moscow| Warsaw
The largest planet in the solar system is| Jupiter| Mars
The planet closest to the sun is| Mercury| Venus
Water is made of hydrogen and| oxygen| nitrogen
The chemical symbol for gold is| Au| Ag
The chemical symbol for sodium is| Na| So
Humans breathe in oxygen and breathe out carbon| dioxide| monoxide
The freezing point of water in degrees Celsius is| 0| 32
There are seven days in a| week| month
A triangle has three| sides| corners only
Two plus two equals| four| five
Ten times ten equals one| hundred| thousand
The square root of nine is| three| four
The first month of the year is| January| March
The season after winter is| spring| autumn
The longest river in Africa is the| Nile| Congo
Mount Everest is located in the| Himalayas| Alps
The Great Wall is located in| China| India
The Eiffel Tower is located in| Paris| London
The Pacific is the largest| ocean| desert
The Sahara is a large| desert| forest
Romeo and Juliet was written by William| Shakespeare| Wordsworth
The theory of relativity was developed by Albert| Einstein| Newton
The first person to walk on the moon was Neil| Armstrong| Aldrin
The author of the Harry Potter books is J. K.| Rowling| Tolkien
The currency of Japan is the| yen| yuan
The currency of the United Kingdom is the| pound| euro
The language mainly spoken in Brazil is| Portuguese| Spanish
The language mainly spoken in Mexico is| Spanish| Portuguese
Bees produce| honey| milk
A spider has eight| legs| wings
The heart pumps| blood| air
The opposite of hot is| cold| warm""".splitlines()]


def read_lines(path: str) -> List[str]:
    """Non-empty lines of a text file; lines starting with '#' are comments."""
    return [l.strip() for l in open(path, encoding="utf8") if l.strip() and not l.startswith("#")]


def stratified(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-balanced subsample of at most n points."""
    if len(labels) <= n:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    per = max(2, n // len(set(labels)))
    return np.sort(np.concatenate([rng.permutation(np.where(labels == c)[0])[:per] for c in dict.fromkeys(labels)]))


def load_data(data_dir: str, n_categories: int, seed: int, holdout: float, pcfg: PipelineConfig) -> Dict[str, Any]:
    """Reads the .txt files. Probe words are split per category into probe (used to analyse and edit) and unseen (evaluation only)."""
    d = data_dir
    rows = [l.split("\t") for l in read_lines(os.path.join(d, "probe_words.txt"))]
    if rows and len(rows[0]) == 1:  # labels in a separate file
        rows = list(zip([r[0] for r in rows], read_lines(os.path.join(d, "probe_labels.txt"))))
    cats_all = list(dict.fromkeys(c for _, c in rows))
    if n_categories > len(cats_all):
        raise ValueError(f"{n_categories} categories requested, probe_words.txt has {len(cats_all)}")
    keep = cats_all[:n_categories]
    rng = np.random.default_rng(seed)
    seen = set()
    probe, unseen = [], []
    for c in keep:
        ws = [w for w, cc in rows if cc == c and not (w.lower() in seen or seen.add(w.lower()))]
        ws = [ws[i] for i in rng.permutation(len(ws))]
        k = max(2, int(round(holdout * len(ws))))
        unseen += [(w, c) for w in ws[:k]]
        probe += [(w, c) for w in ws[k:]]
    held = read_lines(os.path.join(d, "heldout_sentences.txt"))
    return {
        "cats": keep,
        "probe": probe,
        "unseen": unseen,
        "calib": read_lines(os.path.join(d, "calibration_sentences.txt")),
        "guard": held[: pcfg.guard_n],
        "held": held[pcfg.guard_n :],
        "new": read_lines(os.path.join(d, "new_domain_sentences.txt")),
    }


@dataclass
class ProbeData:
    """The probe set and every text collection of one (category count, seed) combination."""

    categories: List[str]  # the first N categories of probe_words.txt
    words: np.ndarray  # probe words (analysed and edited)
    cats: np.ndarray  # their categories
    gen_words: np.ndarray  # unseen words (evaluation only)
    gen_cats: np.ndarray
    calib: List[str]  # calibration sentences
    guard: List[str]  # in-loop guard sentences
    held_eval: List[str]  # held-out sentences used for evaluation only
    new_domain: List[str]  # new-domain sentences (evaluation only)
    cal_texts: List[str]  # calibration sentences plus a sample of probe-word prompts: used to fit the weight edit
    sub: np.ndarray  # stratified subsample of the probe words (topology, intrinsic dimension)
    sub_gen: np.ndarray  # the same for the unseen words

    @classmethod
    def load(cls, data_dir: str, n_categories: int, seed: int, holdout: float, pcfg: PipelineConfig) -> "ProbeData":
        """Read the text files and build every derived collection (the former ``install_data``)."""
        D = load_data(data_dir, n_categories, seed, holdout, pcfg)
        rng = np.random.default_rng(seed)
        words = np.array([w for w, _ in D["probe"]])
        cats = np.array([c for _, c in D["probe"]])
        gen_words = np.array([w for w, _ in D["unseen"]])
        gen_cats = np.array([c for _, c in D["unseen"]])
        pick = rng.permutation(len(words))[: pcfg.calib_words]
        cal_texts = D["calib"] + [pcfg.template.format(w=w) for w in words[pick]]
        data = cls(
            categories=D["cats"],
            words=words,
            cats=cats,
            gen_words=gen_words,
            gen_cats=gen_cats,
            calib=D["calib"],
            guard=D["guard"],
            held_eval=D["held"],
            new_domain=D["new"],
            cal_texts=cal_texts,
            sub=stratified(cats, pcfg.topo_n, seed),
            sub_gen=stratified(gen_cats, pcfg.topo_n, seed),
        )
        print(
            f"[data] {len(data.categories)} categories | {len(data.words)} probe words | {len(data.gen_words)} unseen words | "
            f"{len(D['calib'])} calibration, {len(D['guard'])} guard, {len(D['held'])} held-out, {len(D['new'])} new-domain sentences | seed {seed}"
        )
        return data


# ====================================================================================================
# Benchmarks
# ====================================================================================================
# - **HellaSwag** (validation): choose the plausible continuation of a situation - commonsense reasoning.
# - **ARC-Easy** (test): grade-school science multiple choice.
# - **TruthfulQA MC1** (validation): one true answer among common misconceptions - the standard measure of preferring falsehoods.
#
# Each benchmark is subsampled to `n_bench` examples with a fixed seed and cached in the cache folder. A benchmark that cannot be
# downloaded is reported as missing and left empty; it does not stop the run.


@dataclass
class Benchmark:
    """Benchmark subsets drawn with the run's seed, and the HellaSwag monitor subset (disjoint from the evaluation subset)."""

    tasks: Dict[str, list]  # name -> [(context, [choices], index of the correct choice), ...]
    monitor: list  # per-step check that the model still reasons as before
    seed: int

    @classmethod
    def load(cls, seed: int, cache_dir: str, pcfg: PipelineConfig) -> "Benchmark":
        """Download (or read from the cache) the benchmarks selected in ``pcfg`` and draw the subsets for ``seed``."""
        from datasets import load_dataset

        def get(tries):
            for a, k in tries:
                try:
                    return load_dataset(*a, cache_dir=cache_dir, **k)
                except Exception as e:
                    err = e
            raise err

        def take(ds, extra=0):
            n = min(pcfg.n_bench or len(ds), len(ds))
            return ds.shuffle(seed=seed).select(range(min(n + extra, len(ds))))

        out = {}
        build = {
            "hellaswag": lambda: [
                (r["ctx"], [" " + e for e in r["endings"]], int(r["label"]))
                for r in take(
                    get(
                        [
                            (("Rowan/hellaswag",), {"split": "validation"}),
                            (("Rowan/hellaswag",), {"split": "validation", "revision": "refs/convert/parquet"}),
                        ]
                    ),
                    extra=pcfg.n_monitor,
                )
            ],
            "arc_easy": lambda: [
                (
                    f"Question: {r['question']}\nAnswer:",
                    [" " + t for t in r["choices"]["text"]],
                    r["choices"]["label"].index(r["answerKey"]),
                )
                for r in take(get([(("allenai/ai2_arc", "ARC-Easy"), {"split": "test"})]))
            ],
            "truthfulqa": lambda: [
                (
                    f"Q: {r['question']}\nA:",
                    [" " + c for c in r["mc1_targets"]["choices"]],
                    list(r["mc1_targets"]["labels"]).index(1),
                )
                for r in take(get([(("truthfulqa/truthful_qa", "multiple_choice"), {"split": "validation"})]))
            ],
        }
        for name in pcfg.benchmarks:
            try:
                out[name] = build[name]()
            except Exception as e:
                print(f"[benchmark] {name} unavailable and left out: {str(e)[:120]}")
        hs = out.get("hellaswag", [])
        monitor = []
        if hs and pcfg.n_monitor and len(hs) > pcfg.n_monitor:
            monitor = hs[-pcfg.n_monitor :]
            out["hellaswag"] = hs[: -pcfg.n_monitor]
        print(
            f"[benchmark] seed {seed}: "
            + ", ".join(f"{k} {len(v)}" for k, v in out.items())
            + f" | hellaswag monitor {len(monitor)}"
        )
        return cls(tasks=out, monitor=monitor, seed=seed)
