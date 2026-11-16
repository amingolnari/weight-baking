"""The target model (load, hooks, snapshot, bake, state, scoring) and the analyst model (load, unload, generate)."""

import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Paths, PipelineConfig
from .data import FACTS, Benchmark, ProbeData
from .edits import SvdCache, build_affine
from .prompts import prefix
from .utils import NanReport, ensure, first, free_gpu


def load_lm(path: str, dtype: torch.dtype):
    """Load a causal language model in the given dtype (``dtype=`` on recent transformers, ``torch_dtype=`` on older ones)."""
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)  # older transformers


# ====================================================================================================
# The analyst of one round
# ====================================================================================================
# The analyst of a round is loaded once and proposes the edits for every layer of every target of that round.
# It is one of the pool models (read-only, bfloat16), or the rule-based proposer in the control round (which needs no model).


class AnalystModel:
    """A pool model used read-only as analyst: greedy generation, optional 4-bit weights, explicit load and unload."""

    def __init__(self, entry: Dict[str, Any], paths: Paths, pcfg: PipelineConfig, device: str) -> None:
        self.entry, self.paths, self.pcfg, self.device = entry, paths, pcfg, device
        self.model = None
        self.tok = None

    def load(self) -> "AnalystModel":
        """Make the pool entry the analyst of the current round."""
        a, pcfg = self.entry, self.pcfg
        repo = a.get("analyst_repo", a["repo"])
        d = ensure(repo, self.paths.models_dir if repo == a["repo"] else self.paths.analyst_dir)
        self.tok = AutoTokenizer.from_pretrained(d)
        if "bnb-4bit" in repo:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(d, quantization_config=q, device_map={"": 0}).eval()
        else:
            self.model = (
                load_lm(d, getattr(torch, pcfg.analyst_dtype) if self.device == "cuda" else torch.float32)
                .to(self.device)
                .eval()
            )
        g_ = self.model.generation_config
        g_.max_length = None
        g_.do_sample = False
        g_.temperature = g_.top_p = g_.top_k = None  # max_new_tokens only, greedy
        if g_.pad_token_id is None:
            g_.pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        print(f"[round] arm: {a['arm']} | analyst: {a['name']} ({repo})")
        return self

    def unload(self) -> None:
        """Drop the weights and free the GPU."""
        self.model = self.tok = None
        free_gpu()

    @torch.no_grad()
    def chat(self, msgs: List[Dict[str, str]], L: int) -> str:
        """Greedy reply to the conversation ``msgs``; the reply is forced to start as the JSON object for layer L."""
        if getattr(self.tok, "chat_template", None):
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix(L)
        else:
            text = "\n\n".join(m["content"] for m in msgs) + "\n\n" + prefix(L)  # model without a chat template
        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.pcfg.max_new_tokens)
        return prefix(L) + self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)


# ====================================================================================================
# The target model
# ====================================================================================================
# The first part is the single-model machinery (`snapshot`, hook, bake, state). `ridge_solve` (edits.py) solves the ridge problem in
# its dual form when there are fewer calibration tokens than MLP dimensions, which is the normal case for the larger targets and
# keeps the solve small.
#
# Added for evaluation: `score` (log-likelihood of a continuation), `mc_accuracy` (multiple choice by highest mean token
# log-likelihood, the standard way to run HellaSwag / ARC / TruthfulQA-MC1 on a base language model), `fact_metrics` (the
# factual-preference probe).


class TargetModel:
    """One edited model in full precision, with the run folder, the saved original weights and the log of its layers.

    The edit of a layer is baked into the MLP output projection of its block. ``orig_w`` keeps the weights before each applied edit,
    ``log`` the row of every processed layer, ``base`` the snapshot of the original model (the fixed reference of all limits).
    """

    def __init__(
        self,
        entry: Dict[str, Any],
        out_dir: str,
        edit_dir: str,
        paths: Paths,
        pcfg: PipelineConfig,
        data: ProbeData,
        bench: Benchmark,
        nan: NanReport,
        device: str,
    ) -> None:
        self.entry, self.paths, self.pcfg, self.data, self.bench, self.nan, self.device = (
            entry,
            paths,
            pcfg,
            data,
            bench,
            nan,
            device,
        )
        self.out_dir, self.edit_dir = out_dir, edit_dir
        self.state_dir = os.path.join(out_dir, "state")
        self.model = self.tok = self.blocks = None
        self.layers: List[int] = []
        self.cand: List[int] = []
        self.orig_w: Dict[int, torch.Tensor] = {}
        self.log: Dict[int, Dict[str, Any]] = {}
        self.base: Dict[str, Any] = {}
        self.ref: Optional[List[torch.Tensor]] = None
        self.h_orig: Optional[np.ndarray] = None
        self.svd = SvdCache()

    # ---- paths of this run ----
    def layer_dir(self, L: int) -> str:
        return os.path.join(self.out_dir, f"layer{L:02d}")

    def delta_path(self, L: int) -> str:
        return os.path.join(self.state_dir, f"delta_layer{L}.pt")

    # ---- loading and unloading ----
    def load(self, arm: str, analyst_name: str) -> None:
        """Load the target in full precision for the round of one analyst and point all state at the run folder <arm>/<analyst>/<target>/.
        An interrupted run is resumed: the saved weight deltas are added back and the layer log is read."""
        pcfg, data = self.pcfg, self.data
        os.makedirs(self.state_dir, exist_ok=True)
        d = ensure(self.entry["repo"], self.paths.models_dir)
        self.tok = AutoTokenizer.from_pretrained(d)
        self.model = load_lm(d, getattr(torch, pcfg.target_dtype)).to(self.device).eval()
        self.blocks = next(
            m
            for m in self.model.modules()
            if isinstance(m, torch.nn.ModuleList) and len(m) == self.model.config.num_hidden_layers
        )
        self.layers = list(range(1, len(self.blocks) + 1))
        p = self.tok.encode("a kind of", add_special_tokens=False)  # first token of each category name in context
        self.cand = [self.tok.encode("a kind of " + c, add_special_tokens=False)[len(p)] for c in data.categories]
        self.orig_w.clear()
        self.log.clear()
        self.base = self.snapshot()
        self.base["hsm"] = self.monitor()
        self.h_orig, self.ref = self.base["H"], self.base["lp"]  # original model: fixed reference
        sp, lp_ = os.path.join(self.state_dir, "pipeline_state.json"), os.path.join(self.out_dir, "pipeline_log.csv")
        state = json.load(open(sp)) if pcfg.resume and os.path.exists(sp) and os.path.exists(lp_) else None
        if state and state.get("n_layers") == len(self.layers):
            for L in state["applied"]:
                m = self.lin(L)
                self.orig_w[L] = m.weight.data.clone()
                m.weight.data.add_(torch.load(self.delta_path(L), map_location=self.device).to(m.weight.dtype))
            self.log.update({int(r["layer"]): r for r in pd.read_csv(lp_, keep_default_na=False).to_dict("records")})
        else:
            for f in (sp, lp_, os.path.join(self.out_dir, "target_done.json")):
                if os.path.exists(f):
                    os.remove(f)
        print(
            f"[target] {self.entry['name']} (arm {arm}, analyst {analyst_name}): {len(self.blocks)} layers, d={self.model.config.hidden_size}, "
            f"{pcfg.target_dtype} | {len(self.log)} layers already processed, edits at {sorted(self.orig_w) or 'none'} | "
            f"original ppl {self.base['ppl']:.2f}, cloze {self.base['cloze']:.3f}, hellaswag(monitor) {self.base['hsm']:.3f}"
        )

    def unload(self) -> None:
        """Drop the model and free the GPU."""
        self.model = self.blocks = None
        self.orig_w.clear()
        free_gpu()

    def lin(self, L: int) -> torch.nn.Linear:
        """MLP output projection of block L."""
        return [m for m in self.blocks[L - 1].modules() if isinstance(m, torch.nn.Linear)][-1]

    # ---- forward passes ----
    @torch.no_grad()
    def run(self, text: str):
        """One forward pass with all hidden states."""
        return self.model(**self.tok(text, return_tensors="pt").to(self.device), output_hidden_states=True)

    def _batch(self, texts):
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token  # some tokenizers ship without a pad token
        self.tok.padding_side = "right"  # position 0 stays the first real token (the hook leaves it untouched)
        enc = self.tok(
            list(texts), return_tensors="pt", padding=True, truncation=True, max_length=self.pcfg.max_tokens
        ).to(self.device)
        last = enc["attention_mask"].sum(1) - 1
        return enc, last, torch.arange(len(last), device=last.device)

    @torch.no_grad()
    def feats(self, ws, template: str, bs: int = 64) -> np.ndarray:
        """Hidden state of the last real token for every word at every layer: (words, layers+1, d). Batched, right-padded."""
        out = []
        for i in range(0, len(ws), bs):
            enc, last, idx = self._batch([template.format(w=w) for w in ws[i : i + bs]])
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            out.append(torch.stack([h[idx, last] for h in hs], 1).float().cpu().numpy())
        H = np.concatenate(out)
        if not np.isfinite(H).all():
            raise FloatingPointError("non-finite hidden states (keep target_dtype = 'float32')")
        return H

    @torch.no_grad()
    def cloze_acc(self, ws, labels, template: str, bs: int = 64) -> float:
        """Accuracy of the next-token cloze question restricted to the first tokens of the category names."""
        pred = []
        for i in range(0, len(ws), bs):
            enc, last, idx = self._batch([template.format(w=w) for w in ws[i : i + bs]])
            pred += self.model(**enc).logits[idx, last][:, self.cand].argmax(-1).cpu().tolist()
        return float(np.mean(np.array(self.data.categories)[pred] == labels))

    @torch.no_grad()
    def text_stats(self, texts, states: bool = False, bs: int = 16):
        """Perplexity over ALL texts; log-probabilities only for the first keep_lp texts and hidden states only for the first keep_hs
        (a 500-sentence set would otherwise need gigabytes). Returns (log-probs, hidden states or None, perplexity)."""
        keep_lp, keep_hs = self.pcfg.keep_lp, self.pcfg.keep_hs
        lps, hs, nll, ntok = [], [], 0.0, 0
        for i in range(0, len(texts), bs):
            enc, last, _ = self._batch(texts[i : i + bs])
            need_h = states and i < keep_hs
            out = self.model(**enc, output_hidden_states=need_h)
            for r in range(len(last)):
                k = int(last[r]) + 1
                lp = out.logits[r, :k].float().log_softmax(-1)
                ids = enc["input_ids"][r, :k]
                nll -= lp[:-1].gather(1, ids[1:, None]).sum().item()
                ntok += k - 1
                if i + r < keep_lp:
                    lps.append(lp.cpu())
                if need_h and i + r < keep_hs:
                    hs.append(np.stack([h[r, 1:k].float().cpu().numpy() for h in out.hidden_states], 1))
        return lps, (np.concatenate(hs) if hs else None), float(np.exp(nll / max(ntok, 1)))

    def snapshot(self) -> Dict[str, Any]:
        """Features of the probe words, log-probs / hidden states / perplexity of the guard sentences and cloze accuracy of the current weights."""
        lps, hs, ppl = self.text_stats(self.data.guard, True)
        return {
            "H": self.feats(self.data.words, self.pcfg.template),
            "lp": lps,
            "hs": hs,
            "ppl": ppl,
            "cloze": self.cloze_acc(self.data.words, self.data.cats, self.pcfg.cloze),
        }

    # ---- hook and bake ----
    def make_transform(self, spec: Dict[str, Any], X: np.ndarray) -> Callable[[torch.Tensor], torch.Tensor]:
        """The map h -> h' of a spec, fitted on the layer matrix X, as a torch function."""
        A, b = build_affine(spec, X, self.data.cats, self.svd)
        A_t, b_t = torch.tensor(A, device=self.device), torch.tensor(b, device=self.device)

        def T(h):
            h2 = h @ A_t.to(h.dtype) + b_t.to(h.dtype)
            if spec.get("preserve_norm"):
                h2 = h2 * h.norm(dim=-1, keepdim=True) / h2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return h2

        return T

    def add_hook(self, L: int, T: Callable[[torch.Tensor], torch.Tensor]):
        """Apply T to the output of block L (position 0 untouched). Returns the hook handle; ``handle.remove()`` undoes it."""

        def fn(mod, inp, out):
            h = first(out)
            h2 = torch.cat([h[:, :1], T(h[:, 1:])], 1)  # position 0 untouched
            return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

        return self.blocks[L - 1].register_forward_hook(fn)

    def bake_data(self, L: int, T: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collect MLP activations (keys) and the residual change the target transform asks for, over all calibration sentences plus
        a sample of probe-word prompts."""
        m = self.lin(L)
        keys, outs = [], []
        h1 = m.register_forward_pre_hook(lambda mod, i: keys.append(i[0][0, 1:].double()))
        h2 = self.blocks[L - 1].register_forward_hook(lambda mod, i, o: outs.append(first(o)[0, 1:].double()))
        for t in self.data.cal_texts:
            self.run(t)
        h1.remove()
        h2.remove()
        Kx, Hx = torch.cat(keys), torch.cat(outs)
        return Kx, T(Hx) - Hx

    def save_state(self, checkpoint: bool = False) -> None:
        """Write the applied layers, the pipeline log and (only when asked for and enabled in the config) the full edited model."""
        with open(os.path.join(self.state_dir, "pipeline_state.json"), "w") as f:
            json.dump({"applied": sorted(self.orig_w), "n_layers": len(self.layers)}, f)
        self.nan.clean(pd.DataFrame([self.log[L] for L in sorted(self.log)]), "pipeline_log").to_csv(
            os.path.join(self.out_dir, "pipeline_log.csv"), index=False
        )
        if checkpoint and self.pcfg.save_checkpoint:
            self.model.save_pretrained(self.edit_dir)
            self.tok.save_pretrained(self.edit_dir)

    # ---- scoring ----
    @torch.no_grad()
    def score(self, context: str, continuation: str) -> Tuple[float, float]:
        """(sum, mean) log-probability of the continuation tokens given the context."""
        n_ctx = len(self.tok(context)["input_ids"])
        ids = self.tok(context + continuation, return_tensors="pt")["input_ids"].to(self.device)
        n = max(ids.shape[1] - n_ctx, 1)
        lp = self.model(ids).logits[0, -n - 1 : -1].float().log_softmax(-1)
        ll = lp.gather(1, ids[0, -n:, None])
        return ll.sum().item(), ll.mean().item()

    def mc_accuracy(self, items) -> float:
        """items: (context, [choices], index of the correct choice). Prediction = choice with the highest mean token log-likelihood."""
        return (
            float(np.mean([int(np.argmax([self.score(c, ch)[1] for ch in chs])) == gold for c, chs, gold in items]))
            if items
            else 0.0
        )

    def monitor(self) -> float:
        """HellaSwag accuracy on the monitor subset: the per-step check that the model still reasons as before."""
        return self.mc_accuracy(self.bench.monitor)

    def fact_metrics(self) -> Dict[str, float]:
        """Factual-preference probe: how often (and how confidently) the false completion is preferred."""
        m = np.array(
            [self.score(p, t)[0] - self.score(p, f)[0] for p, t, f in FACTS]
        )  # > 0: the true completion is preferred
        return {
            "facts: accuracy (true preferred)": float((m > 0).mean()),
            "facts: hallucination rate (false preferred)": float((m < 0).mean()),
            "facts: confident hallucination rate (false preferred by > 1 nat)": float((m < -1).mean()),
            "facts: mean truth margin (nats)": float(m.mean()),
        }
