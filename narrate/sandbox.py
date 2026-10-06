"""A2 sandboxed execution tool (§8.2): compute_signal, corr_with, describe, perturb — on training-window
data only, at most 10 calls, every call logged.  Used by narrators at A2 and by judge B3."""
from __future__ import annotations

import json

import numpy as np

from dsl import ParseError, try_parse_any, validate
from verify.inputs import normalize_input
from verify.nudge import nudge_test
from verify.stats import daily_spearman, newey_west_mean

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "compute_signal", "description": "Compute the factor on the training window; returns summary statistics.",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "corr_with", "description": "Mean daily cross-sectional rank correlation with a reference signal.",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}, "reference_name": {"type": "string"}},
                       "required": ["expr", "reference_name"]}}},
    {"type": "function", "function": {
        "name": "describe", "description": "Distribution and coverage of the factor values.",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "perturb", "description": "Raise one input by delta standard deviations and report the share of signal increases.",
        "parameters": {"type": "object", "properties": {"expr": {"type": "string"}, "field": {"type": "string"},
                                                        "delta": {"type": "number"}},
                       "required": ["expr", "field"]}}},
]


class Sandbox:
    def __init__(self, ctx, max_calls: int = 10, window: str = "train"):
        self.ctx = ctx
        self.max_calls = max_calls
        self.window = window
        self.calls: list[dict] = []

    @property
    def exhausted(self) -> bool:
        return len(self.calls) >= self.max_calls

    def _node(self, expr: str):
        node = try_parse_any(expr, strict=False).node
        if not validate(node).ok:
            raise ParseError("invalid formula")
        return node

    def call(self, name: str, args: dict) -> dict:
        if self.exhausted:
            out = {"error": f"tool budget of {self.max_calls} calls exhausted"}
        else:
            try:
                out = getattr(self, f"_t_{name}")(**args)
            except Exception as exc:
                out = {"error": f"{type(exc).__name__}: {exc}"}
        self.calls.append({"name": name, "arguments": args, "result": out})
        return out

    def _t_compute_signal(self, expr: str) -> dict:
        node = self._node(expr)
        rows = self.ctx.rows(self.window)
        f = self.ctx.signal(node)
        ic = daily_spearman(f, self.ctx.fwd(1, "open_t+1"), rows)
        m, se, t = newey_west_mean(ic)
        v = f[rows]
        v = v[np.isfinite(v)]
        return {"mean_rank_ic": round(float(m), 5), "rank_ic_t": round(float(t), 2),
                "mean": float(v.mean()) if v.size else None, "std": float(v.std()) if v.size else None}

    def _t_corr_with(self, expr: str, reference_name: str) -> dict:
        node = self._node(expr)
        refs = self.ctx.references.resolve(reference_name)
        if not refs:
            return {"error": f"unknown reference {reference_name!r}", "available": self.ctx.references.characteristic_names()}
        out = {}
        for name, s in refs:
            r = daily_spearman(self.ctx.signal(node), s * self.ctx.references.signal(name), self.ctx.rows(self.window))
            out[name] = round(float(np.nanmean(r)), 4)
        return {"mean_rank_corr": out}

    def _t_describe(self, expr: str) -> dict:
        node = self._node(expr)
        rows = self.ctx.rows(self.window)
        f = self.ctx.signal(node)[rows]
        mem = self.ctx.panel.member[rows]
        v = f[np.isfinite(f)]
        q = np.quantile(v, [0.01, 0.25, 0.5, 0.75, 0.99]).round(5).tolist() if v.size else []
        return {"quantiles_1_25_50_75_99": q, "coverage": float(np.isfinite(f).sum() / max(1, mem.sum()))}

    def _t_perturb(self, expr: str, field: str, delta: float = 0.5) -> dict:
        node = self._node(expr)
        spec = normalize_input(field)
        v = nudge_test(node, self.ctx, spec, "+", window=self.window, delta_sigma=float(delta),
                       n_contexts=min(100, self.ctx.n_contexts()))
        return {"share_of_increases": v.evidence.get("share_in_claimed_direction"),
                "share_zero": v.evidence.get("share_zero"), "n_contexts": v.evidence.get("n_contexts")}


def tool_message(call_id: str, result: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, default=float)}
