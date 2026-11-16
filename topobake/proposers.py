"""The proposers (arms of a run) and the proposal ladder for one layer.

llm     the analyst LLM only
rule    the rule-based proposer only
hybrid  the LLM first, the rules when the LLM fails

The accepted spec is labelled llm, llm_relaxed, heuristic or heuristic_relaxed. The relaxed-threshold stage can be switched off
(gain-threshold ablation). A stage returns None to hand over to the next one, or the final (spec, hooked, trace, status) tuple.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .edits import heuristic_spec, parse_json, repair
from .geometry import digest, geom
from .models import AnalystModel, TargetModel
from .utils import kl
from .verification import feedback, test_spec

Outcome = Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]], str]


class Proposer:
    """Base class: the escalation ladder is llm stage, rule stage, relaxed-threshold stage. Subclasses switch stages on."""

    arm = ""

    def __init__(
        self,
        analyst: Optional[AnalystModel],
        analyst_name: str,
        pcfg: PipelineConfig,
        system: str,
        descriptors: str = "full",
        relaxed: bool = True,
    ) -> None:
        self.analyst, self.analyst_name, self.pcfg, self.system = analyst, analyst_name, pcfg, system
        self.descriptors, self.relaxed = descriptors, relaxed

    # ---- stages (the base class has none) ----
    def llm_stage(self, target, L, d, X, pre, g0, trace, weak) -> Optional[Outcome]:
        return None

    def rule_stage(self, target, L, d, X, pre, g0, trace, weak) -> Optional[Outcome]:
        return None

    def relaxed_stage(self, target, L, d, X, pre, g0, trace, weak) -> Optional[Outcome]:
        """Retry the strongest rejected-for-gain proposals of each origin with a threshold of 1e-6 (safety limits are not relaxed)."""
        for origin in ("llm", "heuristic"):
            for gain, spec in sorted(weak[origin], key=lambda t: -t[0])[:2]:
                rec = {"stage": f"{origin}_relaxed", "reply": json.dumps(spec["ops"])}
                trace.append(rec)
                acc, hooked = test_spec(target, spec, L, X, pre, g0, 1e-6, rec, [])
                if acc:
                    return self._done(acc, hooked, trace, f"{origin}_relaxed")
                rec["verdict"] = f"rejected ({rec.get('reason')})"
        return None

    @staticmethod
    def _done(spec: Dict[str, Any], hooked: Dict[str, Any], trace: List[Dict[str, Any]], source: str) -> Outcome:
        spec["_sim"]["source"] = source
        return spec, hooked, trace, "ok"

    def propose(
        self, target: TargetModel, L: int, d: Dict[str, pd.DataFrame], X: np.ndarray, pre: Dict[str, Any]
    ) -> Outcome:
        """Proposal ladder for one layer. Returns (spec, hooked snapshot, trace, status); status is "ok", "skipped_by_analyst" or "no_safe_edit"."""
        g0 = geom(X, target.data.cats, self.pcfg.k)
        trace: List[Dict[str, Any]] = []
        weak: Dict[str, list] = {"llm": [], "heuristic": []}
        for stage in (self.llm_stage, self.rule_stage):
            out = stage(target, L, d, X, pre, g0, trace, weak)
            if out is not None:
                return out
        if self.relaxed:
            out = self.relaxed_stage(target, L, d, X, pre, g0, trace, weak)
            if out is not None:
                return out
        return None, None, trace, "no_safe_edit"


class LLMProposer(Proposer):
    """The analyst LLM proposes up to ``max_attempts`` times, each time with the measured feedback of the failed proposal."""

    arm = "llm"

    def _digest(self, target: TargetModel, L: int, d: Dict[str, pd.DataFrame], pre: Dict[str, Any]) -> str:
        applied = [(l, r) for l, r in sorted(target.log.items()) if r["status"] == "applied"]
        return digest(
            L,
            d,
            pre,
            descriptors=self.descriptors,
            n_layers=len(target.blocks),
            cum_kl=kl(target.ref, pre["lp"]),
            base_ppl=target.base["ppl"],
            applied=applied,
            pcfg=self.pcfg,
        )

    def llm_stage(self, target, L, d, X, pre, g0, trace, weak) -> Optional[Outcome]:
        pcfg = self.pcfg
        msgs = [{"role": "user", "content": self.system + "\n\n" + self._digest(target, L, d, pre)}]
        for a in range(pcfg.max_attempts):
            raw = self.analyst.chat(msgs, L)
            rec = {"stage": "llm", "attempt": a + 1, "reply": raw}
            trace.append(rec)
            try:
                spec = parse_json(raw)
                no_edit = isinstance(spec, dict) and spec.get("ops") == []
                spec, rec["repairs"] = repair(spec, L, X, target.data.categories, pcfg.max_ops)
                if no_edit:
                    rec["verdict"] = "analyst: no edit"
                    return None, None, trace, "skipped_by_analyst"
                if not spec["ops"]:
                    raise ValueError("the reply contains no usable operation")
            except Exception as e:
                rec["reason"] = "invalid"
                fb = f"INVALID REPLY: {e}. Use only the allowed operations and category names."
            else:
                acc, hooked = test_spec(target, spec, L, X, pre, g0, pcfg.min_gain, rec, weak["llm"])
                if acc:
                    return self._done(acc, hooked, trace, "llm")
                fb = feedback(rec, L, g0, pcfg)
            rec["verdict"] = fb
            msgs += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": fb + " Return a corrected JSON object only."},
            ]
        return None


class RuleProposer(Proposer):
    """Rule-based proposals: the 3, 2, then 1 most conflated category pairs, each stretched by an adaptive factor."""

    arm = "rule"

    def rule_stage(self, target, L, d, X, pre, g0, trace, weak) -> Optional[Outcome]:
        for n in (3, 2, 1):
            spec = heuristic_spec(L, d, n)
            rec = {"stage": "heuristic", "reply": json.dumps(spec["ops"])}
            trace.append(rec)
            acc, hooked = test_spec(target, spec, L, X, pre, g0, self.pcfg.min_gain, rec, weak["heuristic"])
            if acc:
                return self._done(acc, hooked, trace, "heuristic")
            rec["verdict"] = f"rejected ({rec.get('reason')})"
        return None


class HybridProposer(LLMProposer, RuleProposer):
    """The LLM stage first, the rule stage when the LLM fails (both stages are inherited)."""

    arm = "hybrid"
