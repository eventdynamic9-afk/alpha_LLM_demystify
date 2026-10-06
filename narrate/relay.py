"""File relay for LLM calls answered by an external agent (§4.4 fallback when no endpoint is reachable).

A :class:`RelayClient` call is keyed by a hash of (model, messages, tools, temperature, max_tokens,
seed).  When ``<relay_dir>/responses/<key>.txt`` exists its content is the model's reply; otherwise
the request is rendered to ``<relay_dir>/requests/<key>.txt`` (plus a ``.json`` with metadata) and
:class:`PendingResponse` is raised.  Every pipeline stage catches it, skips the item and reports the
number of pending requests; re-running the stage after the requests are answered completes it.

Each request is meant to be answered in a fresh context by one agent that reads only that file and
writes only the response file (no memory, no retrieval, no other tools: §8.5).  Tool calls (A2, B3)
use a text protocol: the reply contains one ``TOOL_CALL: {"name": ..., "arguments": {...}}`` line per
call and nothing else; the stage executes the calls and issues the next turn as a new request whose
messages carry the full conversation, exactly as a stateless chat-completions API would.  Decoding
parameters (temperature, seed) are recorded in the request but cannot be enforced on the agent; the
k samples of a cell are independent fresh-context requests.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

TOOL_PROTOCOL = (
    "You may call the tools listed below. To call tools, reply with one line per call, exactly in the form\n"
    'TOOL_CALL: {"name": "<tool name>", "arguments": {<JSON arguments>}}\n'
    "and nothing else in that reply. Tool results come back as `tool` messages. When you are done with "
    "tools, reply with your final answer (no TOOL_CALL lines)."
)
_CALL = re.compile(r"^\s*TOOL_CALL:\s*(\{.*\})\s*$", re.M)


class PendingResponse(Exception):
    """A relay request was written (or already exists) but has not been answered yet."""

    def __init__(self, key: str, path: Path):
        super().__init__(f"relay request {key} pending: {path}")
        self.key = key
        self.path = path


def request_key(model_id: str, messages: list, tools, temperature: float, max_tokens: int, seed) -> str:
    payload = json.dumps({"model": model_id, "messages": messages, "tools": tools, "temperature": round(temperature, 3),
                          "max_tokens": max_tokens, "seed": seed}, sort_keys=True, default=str)
    return f"{model_id}-{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def _tool_lines(tools: list) -> str:
    out = []
    for t in tools:
        fn = t.get("function", t)
        props = fn.get("parameters", {}).get("properties", {})
        args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
        out.append(f"- {fn['name']}({args}): {fn.get('description', '')}")
    return "\n".join(out)


def render_request(messages: list, tools: list | None = None, max_tokens: int | None = None) -> str:
    """Plain-text rendering of a chat request: one block per message, in order."""
    parts = []
    for m in messages:
        role = m["role"]
        if role == "assistant" and m.get("tool_calls"):
            calls = "\n".join("TOOL_CALL: " + json.dumps({"name": c["function"]["name"],
                                                          "arguments": json.loads(c["function"]["arguments"] or "{}")})
                              for c in m["tool_calls"])
            body = ((m.get("content") or "").strip() + "\n" + calls).strip()
            parts.append(f"=== assistant ===\n{body}")
        elif role == "tool":
            parts.append(f"=== tool ({m.get('tool_call_id', '')}) ===\n{m.get('content', '')}")
        else:
            parts.append(f"=== {role} ===\n{m.get('content', '')}")
    if tools:
        parts.insert(1 if messages and messages[0]["role"] == "system" else 0,
                     f"=== tools ===\n{TOOL_PROTOCOL}\n\n{_tool_lines(tools)}")
    head = "Reply as the assistant to the conversation below. Write only the assistant's next message."
    if max_tokens:
        head += f" Keep it within about {int(max_tokens * 0.75)} words."
    return head + "\n\n" + "\n\n".join(parts) + "\n\n=== assistant ===\n"


def parse_reply(text: str) -> tuple[str, list[dict]]:
    """Split a relay reply into (text, tool_calls). Tool-call ids are deterministic (call_0, call_1...)."""
    calls = []
    for i, m in enumerate(_CALL.finditer(text or "")):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and d.get("name"):
            calls.append({"id": f"call_{i}", "name": d["name"], "arguments": d.get("arguments") or {}})
    rest = _CALL.sub("", text or "").strip() if calls else (text or "").strip()
    return rest, calls


class Relay:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.req = self.root / "requests"
        self.resp = self.root / "responses"

    def response(self, key: str) -> str | None:
        p = self.resp / f"{key}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def write_request(self, key: str, messages: list, tools, meta: dict, max_tokens: int | None) -> Path:
        self.req.mkdir(parents=True, exist_ok=True)
        p = self.req / f"{key}.txt"
        if not p.exists():
            p.write_text(render_request(messages, tools, max_tokens), encoding="utf-8")
            (self.req / f"{key}.json").write_text(json.dumps({"key": key, **meta}, default=str), encoding="utf-8")
        return p

    def pending(self) -> list[dict]:
        out = []
        for j in sorted(self.req.glob("*.json")) if self.req.exists() else []:
            meta = json.loads(j.read_text(encoding="utf-8"))
            if not (self.resp / f"{meta['key']}.txt").exists():
                out.append({**meta, "request": str(self.req / f"{meta['key']}.txt"),
                            "response": str(self.resp / f"{meta['key']}.txt")})
        return out

    def put(self, key: str, text: str) -> Path:
        self.resp.mkdir(parents=True, exist_ok=True)
        p = self.resp / f"{key}.txt"
        p.write_text(text, encoding="utf-8")
        return p


def import_journal(relay: Relay, journal: str | Path) -> dict:
    """Recover replies that answering agents returned as text instead of writing the response file.

    Agent labels are ``<tier>:<last 6 characters of the key>``; a reply is imported only when exactly one
    pending request matches the label, and replies that are just an acknowledgement are ignored.
    """
    labels, results = {}, []
    with open(journal, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("type") == "started":
                labels[d["agentId"]] = d.get("label", "")
            elif d.get("type") == "result":
                results.append(d)
    pend = relay.pending()
    imported, skipped = [], []
    for d in results:
        text = d.get("result")
        text = text if isinstance(text, str) else json.dumps(text)
        if text.strip().strip(".").lower() in ("done", "") or len(text.split()) < 5:
            continue
        tier, _, suffix = labels.get(d.get("agentId"), "").partition(":")
        match = [p for p in pend if p["key"].endswith(suffix) and p["key"].split("-")[1] == tier]
        if len(match) == 1:
            relay.put(match[0]["key"], text.strip())
            imported.append(match[0]["key"])
        else:
            skipped.append({"label": labels.get(d.get("agentId")), "matches": len(match)})
    return {"imported": imported, "skipped": skipped}
