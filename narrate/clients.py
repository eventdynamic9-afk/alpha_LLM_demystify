"""LLM clients (§4.4).

* :class:`OpenAICompatibleClient` — any ``/v1/chat/completions`` endpoint: local vLLM / llama.cpp /
  Ollama / LM Studio servers (Kaggle, Colab, own GPU), and free or paid API tiers that expose the
  OpenAI-compatible protocol (OpenRouter, Groq, Mistral, Cerebras, Gemini's compatibility endpoint...).
  Uses only the standard library.
* :class:`RelayClient` — file relay answered by an external agent in a fresh context
  (:mod:`narrate.relay`), for runs where no endpoint is reachable.
* :class:`MockClient` — deterministic offline simulator used by tests and the smoke pipeline.  It
  "reads" the formula with the static analysers, makes misreadings at a configured rate, recalls the
  base formula of a recognised public alpha (anchoring), follows misleading labels at a configured
  rate and adds hallucinated behavioral claims.  It is never used for results.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    tool_calls: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    fingerprint: str | None = None
    model: str = ""
    finish_reason: str | None = None
    raw: dict | None = None
    reasoning: str | None = None                       # reasoning text when the endpoint exposes it (§4.3)
    params: dict = field(default_factory=dict)         # request parameters actually sent (minus messages/tools)


PROVENANCE_FIELDS = ("provider", "model_string", "base_url", "weights_sha256", "quantization", "engine",
                     "engine_version", "agent_tier", "training_cutoff")


class LLMClient:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model_id = cfg["id"]
        self.model_string = cfg.get("model_string", cfg["id"])
        self.family = cfg.get("family", "")
        # §8.3: reasoning models run at their default reasoning setting, recorded with every call
        self.reasoning_setting = cfg.get("reasoning_setting") or ("default" if cfg.get("reasoning") else None)

    def provenance(self) -> dict:
        """Per-model version pinning fields (§4.1) written into every call record."""
        return {"model_id": self.model_id, **{k: self.cfg.get(k) for k in PROVENANCE_FIELDS if k in self.cfg},
                "reasoning_setting": self.reasoning_setting}

    def complete(self, messages: list[dict], tools: list | None = None, temperature: float = 0.7,
                 max_tokens: int = 700, seed: int | None = None, tool_choice: str | None = None,
                 reasoning_effort: str | None = None) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.base_url = cfg["base_url"].rstrip("/")
        key_env = cfg.get("api_key_env")
        self.api_key = os.environ.get(key_env, "") if key_env else os.environ.get("OPENAI_API_KEY", "none")
        self.timeout = int(cfg.get("timeout", 300))
        self.retries = int(cfg.get("retries", 4))

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=700, seed=None, tool_choice=None,
                 reasoning_effort=None) -> LLMResponse:
        body = {"model": self.model_string, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        effort = reasoning_effort or (self.reasoning_setting if self.reasoning_setting not in (None, "default") else None)
        if effort:                                    # §14 reasoning-effort ablation / a non-default pinned setting
            body[self.cfg.get("reasoning_effort_key", "reasoning_effort")] = effort
        body.update(self.cfg.get("extra_body", {}))
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        delay = 2
        t0 = time.time()
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                break
            except Exception:
                if attempt == self.retries:
                    raise
                time.sleep(delay)
                delay *= 2
        ch = data["choices"][0]
        msg = ch.get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_raw": fn.get("arguments")}
            calls.append({"id": tc.get("id"), "name": fn.get("name"), "arguments": args})
        usage = data.get("usage") or {}
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return LLMResponse(msg.get("content") or "", calls,
                           {"in": usage.get("prompt_tokens"), "out": usage.get("completion_tokens"),
                            "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")},
                           time.time() - t0, data.get("system_fingerprint"), data.get("model", self.model_string),
                           ch.get("finish_reason"), data, reasoning if isinstance(reasoning, str) else
                           (json.dumps(reasoning) if reasoning else None),
                           {k: v for k, v in body.items() if k not in ("messages", "tools")})


class RelayClient(LLMClient):
    """Writes each unanswered request to the relay directory and raises :class:`PendingResponse`."""

    def __init__(self, cfg: dict):
        from .relay import Relay

        super().__init__(cfg)
        self.relay = Relay(os.environ.get("RELAY_DIR") or cfg.get("relay_dir") or "runs/relay")

    def complete(self, messages, tools=None, temperature=0.7, max_tokens=700, seed=None, tool_choice=None,
                 reasoning_effort=None) -> LLMResponse:
        from .relay import PendingResponse, parse_reply, request_key

        if tool_choice == "none":                     # final turn with tools disabled: no tool protocol shown
            tools = None
        key = request_key(self.model_id, messages, tools, temperature, max_tokens, seed)
        if reasoning_effort:                          # keeps effort-ablation requests distinct from primary ones
            key = request_key(self.model_id, messages, tools, temperature, max_tokens,
                              f"{seed}|reasoning_effort={reasoning_effort}")
        params = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens, "has_tools": bool(tools)}
        params.update({k: v for k, v in (("tool_choice", tool_choice), ("reasoning_effort", reasoning_effort)) if v})
        reply = self.relay.response(key)
        if reply is None:
            path = self.relay.write_request(key, messages, tools,
                                            {"model": self.model_id, "agent_tier": self.cfg.get("agent_tier"), **params},
                                            max_tokens)
            raise PendingResponse(key, path)
        text, calls = parse_reply(reply) if tools else (reply.strip(), [])
        return LLMResponse(text, calls, {"in": None, "out": len(reply.split())}, 0.0, key, self.model_string,
                           "tool_calls" if calls else "stop", params=params)


# =============================================================================== mock simulator
_FIELD_WORDS = {"close": "the closing price", "open": "the opening price", "high": "the daily high",
                "low": "the daily low", "vwap": "the VWAP", "volume": "trading volume", "amount": "dollar volume"}


def _skeleton(node) -> str:
    """Structure with windows, constants, signs and field identities removed (recall key)."""
    from dsl import walk

    parts = []
    for n in walk(node):
        if n.is_field:
            parts.append("F")
        elif n.is_const:
            continue
        elif n.op in ("Neg",):
            continue
        else:
            parts.append(n.op)
    return "-".join(parts)


class MockClient(LLMClient):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        m = cfg.get("mock", {})
        self.misread = float(m.get("misread_rate", 0.2))
        self.recall = float(m.get("recall_rate", 0.3))
        self.halluc = float(m.get("hallucinate_rate", 0.2))
        self._skeletons = None

    # ------------------------------------------------------------------ dispatch on prompt type
    def complete(self, messages, tools=None, temperature=0.7, max_tokens=700, seed=None, tool_choice=None,
                 reasoning_effort=None) -> LLMResponse:
        prompt = "\n".join(m.get("content") or "" for m in messages if m["role"] in ("system", "user"))
        h = int(hashlib.sha256((self.model_id + prompt + str(seed) + str(temperature)).encode()).hexdigest()[:12], 16)
        rng = random.Random(h)
        t0 = time.time()
        if "Split the rationale into atomic claims" in prompt:
            text = self._parse(prompt)
        elif "Rate how consistent the factor DESCRIPTION" in prompt:
            text = self._judge_b1(prompt, rng)
        elif "You are checking a single claim" in prompt:
            text = json.dumps({"verdict": rng.choice(["TRUE", "TRUE", "FALSE", "CANT_TELL"]), "reason": "mock"})
        elif "Reconstruct the factor formula" in prompt:
            text = self._reconstruct(prompt, rng)
        elif "Write ONE" in prompt or "Refine the formula" in prompt:
            text = self._author(prompt, rng)
        else:
            text = self._narrate(prompt, rng)
        return LLMResponse(text, [], {"in": len(prompt) // 4, "out": len(text) // 4}, time.time() - t0,
                           f"mock-{self.model_id}", self.model_string, "stop",
                           params={"temperature": temperature, "max_tokens": max_tokens, "seed": seed,
                                   "tool_choice": tool_choice or ("auto" if tools else None),
                                   "reasoning_effort": reasoning_effort})

    # ------------------------------------------------------------------ narration
    def _recall_base(self, node):
        from dsl import canonical_equal
        from pools.library import base_set

        if self._skeletons is None:
            self._skeletons = {}
            for lf in base_set():
                self._skeletons.setdefault(_skeleton(lf.node), lf)
        for lf in base_set():
            if canonical_equal(lf.node, node):
                return lf
        return self._skeletons.get(_skeleton(node))

    def _read_formula(self, prompt: str):
        from dsl import try_parse_any

        m = re.search(r"Factor:\s*(.+?)\n(?:\s*\n|Name:|In your rationale|Explain|Diagnostics|Tools:)", prompt + "\n\n", re.S)
        if not m:
            return None
        src = m.group(1).strip()
        from dsl.fields import FIELDS

        by_desc = {f.description: name for name, f in FIELDS.items()}
        for lab, desc in re.findall(r"\b(x\d+): ([^;\n]+)", prompt):
            fld = by_desc.get(desc.strip())
            if fld:
                src = re.sub(rf"\b{lab}\b", f"${fld}", src)
        try:
            return try_parse_any(src, strict=False).node
        except Exception:
            try:
                from dsl import parse

                return parse(src.replace("\n", ";"))
            except Exception:
                return None

    def _narrate(self, prompt: str, rng: random.Random) -> str:
        from dsl import effective_lookback, walk
        from dsl.monotonicity import NEG, POS, current_value_direction
        from dsl.operators import OPS

        node = self._read_formula(prompt)
        if node is None:
            return "I cannot interpret this formula."
        sentences = []
        label = re.search(r"Name:\s*(.+)", prompt)
        src_node = node
        recalled = None
        if rng.random() < self.recall:
            recalled = self._recall_base(node)
            if recalled is not None:
                src_node = recalled.node            # anchoring: describe the remembered formula
                if rng.random() < 0.5:
                    sentences.append(f"This is {recalled.display_name}.")
        fields = sorted({n.name for n in walk(src_node) if n.is_field})
        sentences.append("The factor is driven by " + ", ".join(_FIELD_WORDS[f] for f in fields) + ".")
        for f in fields[:2]:
            d = current_value_direction(src_node, f)
            if d in (POS, NEG):
                up = d == POS
                if rng.random() < self.misread:
                    up = not up
                sentences.append(f"A higher {f} today raises the factor value." if up
                                 else f"A higher {f} today lowers the factor value.")
        L = effective_lookback(src_node)
        if rng.random() < self.misread:
            L = L + rng.choice([5, 10, 20])
        sentences.append(f"It uses a {L + 1}-day window.")
        sentences.append("It is a short-term signal." if L + 1 <= 21 else
                         "It is a medium-term signal." if L + 1 <= 126 else "It is a long-term signal.")
        if any(not n.is_leaf and OPS[n.op].kind == "xs" for n in walk(src_node)):
            sentences.append("It ranks stocks against each other.")
        if label and rng.random() < self.recall:
            txt = label.group(1).replace(" factor", "")
            sentences.append(f"It captures {txt}.")
        elif rng.random() < self.halluc:
            sentences.append(rng.choice(["It captures short-term reversal.", "It resembles momentum.",
                                         "It has a low-volatility tilt.", "It is independent of momentum."]))
        sign = rng.choice(["higher", "lower"])
        sentences.append(f"High values predict {sign} future returns over the next day.")
        if rng.random() < self.halluc:
            sentences.append("It is novel and not explained by existing factors.")
        if rng.random() < 0.5:
            sentences.append("It works best in volatile markets.")
        if rng.random() < 0.6:
            sentences.append("This is because investors overreact to recent news.")
        return " ".join(sentences)

    # ------------------------------------------------------------------ other roles
    def _parse(self, prompt: str) -> str:
        from parse.rules import extract_claims

        m = re.search(r"Rationale id: (\S+)\nRationale:\n<<<\n(.*)\n>>>", prompt, re.S)
        rid, text = (m.group(1), m.group(2)) if m else ("unknown", prompt)
        claims = extract_claims(text, rid)
        return json.dumps({"rationale_id": rid, "claims": claims})

    def _judge_b1(self, prompt: str, rng: random.Random) -> str:
        has_h = '"c1"' in prompt or "field \"c1\"" in prompt
        out = {"c2": round(rng.uniform(0.6, 1.0), 2)}
        if has_h:
            out["c1"] = round(rng.uniform(0.6, 1.0), 2)
        out["explanation"] = "mock judgement"
        return json.dumps(out)

    def _reconstruct(self, prompt: str, rng: random.Random) -> str:
        from dsl import to_qlib
        from dsl.random_trees import random_tree

        m = re.search(r"window[^0-9]*(\d+)", prompt)
        n = int(m.group(1)) - 1 if m else 5
        if "lower" in prompt or "reversal" in prompt:
            return f"FORMULA: -1*($close/Ref($close, {max(1, n)})-1)"
        return "FORMULA: " + to_qlib(random_tree(rng, max_depth=3))

    def _author(self, prompt: str, rng: random.Random) -> str:
        from dsl import to_qlib
        from dsl.random_trees import random_tree

        t = random_tree(rng, max_depth=rng.randint(3, 5))
        hyp = ""
        if "Step 1" in prompt:
            d = re.search(r"Research direction:\s*(.+)", prompt)
            topic = d.group(1).strip() if d else "price dynamics"
            hyp = (f"Observation: {topic} varies across stocks.\nMechanism: investors underreact to {topic}.\n"
                   f"Specification: measure {topic} with recent prices and volume.\n")
        return f"{hyp}FORMULA: {to_qlib(t)}"


def get_client(cfg: dict) -> LLMClient:
    prov = cfg.get("provider", "openai_compatible")
    if prov == "mock":
        return MockClient(cfg)
    if prov == "relay":
        return RelayClient(cfg)
    if prov in ("openai_compatible", "local", "openrouter", "groq", "mistral", "cerebras", "gemini_openai"):
        if not cfg.get("base_url") or cfg.get("base_url") == "TO_FILL":
            raise ValueError(f"model {cfg['id']}: base_url must be set on the run date")
        return OpenAICompatibleClient(cfg)
    raise ValueError(f"unknown provider {prov!r}")
