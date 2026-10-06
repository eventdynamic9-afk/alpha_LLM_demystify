"""Narration runner (§8): fresh context per narration, no memory or retrieval; refusals, empty and
off-topic outputs logged and counted and re-sampled at most once; A2 tool loop with the sandbox.

A2 (§8.2): once the tool budget is used, the tool results plus a fixed "budget exhausted" message are
sent and one final turn is requested with tools disabled (``tool_choice="none"``), so models that check
the most still return a rationale; an attempt that ends without text after exhaustion is recorded as
``tool_budget_exhausted`` (not ``empty``) and the tool logs of every attempt are kept.  Every call logs
the request parameters actually sent, the reasoning setting (§8.3), reasoning text when exposed and the
model's provenance fields (§4.1, §4.3)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from configs import prompt, study, template_sha256
from dsl import parse
from pools.records import now_iso

from .clients import get_client
from .logger import CallLogger, ResponseCache, prompt_hash
from .plan import Cell
from .prompts import build_messages
from .relay import PendingResponse
from .sandbox import TOOL_SCHEMAS, Sandbox, tool_message

_REFUSAL = re.compile(r"\b(i can(?:no|')t|i am unable|i'm unable|i'm sorry|as an ai|i cannot help)\b", re.I)


def classify_output(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty"
    if _REFUSAL.search(t[:200]) and len(t.split()) < 60:
        return "refusal"
    if len(t.split()) < 15:
        return "off_topic"
    return "ok"


class NarrationRunner:
    def __init__(self, models_cfg: dict, records: dict[str, dict], run_dir: str | Path, ctx=None,
                 market: str = "CN", max_tokens: int | None = None):
        self.cfg = models_cfg
        self.records = records
        self.run_dir = Path(run_dir)
        self.ctx = ctx
        self.market = market
        self.clients = {m["id"]: get_client(m) for m in models_cfg["models"]}
        self.logger = CallLogger(self.run_dir / "calls.jsonl")
        self.cache = ResponseCache(self.run_dir / "cache.sqlite")
        self.max_tokens = max_tokens or models_cfg.get("decoding", {}).get("max_tokens", 700)
        self._diag: dict[str, str] = {}

    # ------------------------------------------------------------------ helpers
    def _diagnostics(self, rec: dict) -> str:
        if rec["formula_id"] not in self._diag:
            from .diagnostics import diagnostics, diagnostics_table

            self._diag[rec["formula_id"]] = diagnostics_table(diagnostics(parse(rec["dsl"]), self.ctx))
        return self._diag[rec["formula_id"]]

    def _call(self, model: str, messages: list, tools, temperature: float, sample: int, role: str,
              meta: dict, tool_choice: str | None = None, reasoning_effort: str | None = None) -> dict:
        ph = prompt_hash(messages, tools)
        if tool_choice or reasoning_effort:          # non-default request options are part of the cache key
            ph = hashlib.sha256(f"{ph}|tool_choice={tool_choice}|reasoning_effort={reasoning_effort}".encode()).hexdigest()
        cached = self.cache.get(model, ph, sample, temperature)
        if cached is not None:
            return cached
        client = self.clients[model]
        seed = None if sample < 0 else sample
        kw = {k: v for k, v in (("tool_choice", tool_choice), ("reasoning_effort", reasoning_effort)) if v}
        r = client.complete(messages, tools, temperature, self.max_tokens, seed, **kw)
        resp = {"text": r.text, "tool_calls": r.tool_calls, "usage": r.usage, "latency_s": r.latency_s,
                "fingerprint": r.fingerprint, "model": r.model, "finish_reason": r.finish_reason,
                "reasoning": getattr(r, "reasoning", None)}
        self.cache.put(model, ph, sample, temperature, resp)
        params = {"temperature": temperature, "max_tokens": self.max_tokens, "seed": seed,
                  "tool_choice": tool_choice or ("auto" if tools else None),
                  "reasoning_setting": reasoning_effort or getattr(client, "reasoning_setting", None)}
        prov = client.provenance() if hasattr(client, "provenance") else {"model_string": client.model_string}
        self.logger.log(role=role, model=model, model_string=client.model_string, prompt_hash=ph,
                        request={"messages": messages, "tools": tools}, response=resp, params=params,
                        request_params=getattr(r, "params", None) or {}, provenance=prov,
                        timestamp=now_iso(), **meta)
        return resp

    # ------------------------------------------------------------------ one narration
    def narrate(self, cell: Cell) -> dict:
        rec = self.records[cell.formula_id]
        key = f"{cell.model}|{cell.formula_id}|{cell.access}|{cell.variant}"
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)   # stable glossary/field order
        tw = None
        diag = None
        if cell.access == "A1":
            diag = self._diagnostics(rec)
            tw = "{} to {}".format(*self.ctx.windows["train"]) if self.ctx else None
        messages, meta = build_messages(rec, cell.access, cell.variant, self.market, seed, diag, tw)
        effort = getattr(cell, "reasoning_effort", None)
        tools = TOOL_SCHEMAS if cell.access == "A2" else None
        attempts = []
        status = "ok"
        exhausted = False
        call_meta = {"formula_id": cell.formula_id, "access": cell.access, "prompt_variant": cell.variant, **meta}
        for attempt in range(2):                     # at most one re-sample
            sample = cell.sample_idx if attempt == 0 else cell.sample_idx + 1000
            msgs = list(messages)
            sandbox = Sandbox(self.ctx, study()["access"]["A2_max_tool_calls"]) if tools and self.ctx else None
            resp = self._call(cell.model, msgs, tools, cell.temperature, sample, "narrator", call_meta,
                              reasoning_effort=effort)
            tool_log = []
            exhausted = False
            while sandbox is not None and resp.get("tool_calls"):
                msgs.append({"role": "assistant", "content": resp.get("text") or "",
                             "tool_calls": [{"id": tc["id"], "type": "function",
                                             "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                                            for tc in resp["tool_calls"]]})
                for tc in resp["tool_calls"]:
                    out = sandbox.call(tc["name"], tc.get("arguments") or {})
                    tool_log.append({"name": tc["name"], "arguments": tc.get("arguments"), "result": out})
                    msgs.append(tool_message(tc["id"], out))
                if sandbox.exhausted:                # §8.2 budget used: one final turn with tools disabled
                    exhausted = True
                    note = prompt("narrator_a2_budget_exhausted")
                    msgs.append({"role": "user", "content": note})
                    fm = {**call_meta, "template_sha256": {**meta["template_sha256"],
                                                           "a2_budget_exhausted": template_sha256(note)}}
                    resp = self._call(cell.model, msgs, tools, cell.temperature, sample, "narrator_final_turn", fm,
                                      tool_choice="none", reasoning_effort=effort)
                    break
                resp = self._call(cell.model, msgs, tools, cell.temperature, sample, "narrator_tool_turn", call_meta,
                                  reasoning_effort=effort)
            status = classify_output(resp.get("text", ""))
            if exhausted and status == "empty":
                status = "tool_budget_exhausted"
            attempts.append({"sample": sample, "status": status, "tool_budget_exhausted": exhausted,
                             "tool_calls": tool_log})
            if status == "ok":
                break
        text = resp.get("text", "")
        if rec.get("pool") == "P1" and rec.get("hypothesis"):
            text = f"{rec['hypothesis'].strip()}\n\n{text}"
        client = self.clients[cell.model]
        return {"rationale_id": cell.rationale_id, "formula_id": cell.formula_id, "model": cell.model,
                "model_version": client.model_string, "served_model": resp.get("model"), "family": client.family,
                "access": cell.access, "prompt_variant": cell.variant, "sample_idx": cell.sample_idx,
                "temperature": cell.temperature, "reasoning_setting": effort or getattr(client, "reasoning_setting", None),
                "text": text, "tool_calls": tool_log, "tool_budget_exhausted": exhausted,
                "tokens": {"in": resp.get("usage", {}).get("in"), "out": resp.get("usage", {}).get("out")},
                "timestamp": now_iso(), "arm": cell.arm, "pool": cell.pool, "cross_narration": cell.cross,
                "author_model": rec.get("author_model"), "status": status, "attempts": attempts,
                "fingerprint": resp.get("fingerprint")}

    def run(self, cells: list[Cell], out_path: str | Path | None = None) -> list[dict]:
        out_path = Path(out_path or self.run_dir / "rationales.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out_path.exists():
            with open(out_path, encoding="utf-8") as fh:
                done = {json.loads(line)["rationale_id"] for line in fh if line.strip()}
        results = []
        self.pending = 0
        with open(out_path, "a", encoding="utf-8") as fh:
            for c in cells:
                if c.rationale_id in done:
                    continue
                try:
                    r = self.narrate(c)
                except PendingResponse:              # relay request written; re-run after it is answered
                    self.pending += 1
                    continue
                fh.write(json.dumps(r, default=str) + "\n")
                results.append(r)
        return results
