"""Normalization of the ``input`` argument of DEPENDS_ON / SIGN / MONO claims (§9.5)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from configs import codebook
from dsl.fields import FIELDS


@dataclass(frozen=True)
class InputSpec:
    kind: str              # field | ret | abn_vol | volatility | range | intraday | out_of_universe | unknown
    field: str | None = None
    d: int = 1             # horizon for returns / window for volatility
    raw: str = ""

    @property
    def fields(self) -> tuple[str, ...]:
        """Panel fields the quantity is built from (for static dependency checks)."""
        if self.kind == "field":
            return (self.field,)
        if self.kind in ("ret", "volatility"):
            return ("close",)
        if self.kind == "abn_vol":
            return ("volume",)
        if self.kind == "range":
            return ("high", "low")
        if self.kind == "intraday":
            return ("close", "open")
        return ()


_ALIASES = {"price": "close", "prices": "close", "closing price": "close", "share price": "close",
            "stock price": "close", "dollar volume": "amount", "turnover": "volume", "trading volume": "volume",
            "trading activity": "volume", "volume-weighted average price": "vwap"}


def normalize_input(name: str) -> InputSpec:
    raw = str(name)
    s = raw.strip().lower().replace("$", "")
    if s in FIELDS:
        return InputSpec("field", s, raw=raw)
    if s in _ALIASES:
        return InputSpec("field", _ALIASES[s], raw=raw)
    m = re.fullmatch(r"(?:ret|return|returns)(?:[_\s-]?(\d+)\s*d?)?", s)
    if m:
        return InputSpec("ret", None, int(m.group(1) or 1), raw)
    m = re.fullmatch(r"(?:past|recent|last)?\s*(\d+)[-\s]?day returns?", s)
    if m:
        return InputSpec("ret", None, int(m.group(1)), raw)
    if s in ("recent return", "recent returns", "past return", "past returns", "price change", "momentum"):
        return InputSpec("ret", None, 5 if "recent" in s or "change" in s else 21, raw)
    if s in ("abn_vol", "abnormal volume", "volume surprise", "volume shock", "volume spike"):
        return InputSpec("abn_vol", "volume", raw=raw)
    m = re.fullmatch(r"(?:volatility|vol)(?:[_\s-]?(\d+)\s*d?)?", s)
    if m:
        return InputSpec("volatility", None, int(m.group(1) or 20), raw)
    if s in ("range", "daily range", "high-low range", "trading range"):
        return InputSpec("range", None, raw=raw)
    if s in ("intraday", "intraday return", "close-open", "intraday move"):
        return InputSpec("intraday", None, raw=raw)
    for f, words in codebook().get("field_lexicon", {}).items():
        if s in words:
            if f == "returns":
                return InputSpec("ret", None, 1, raw)
            return InputSpec("field", f, raw=raw)
    oou = codebook().get("out_of_universe_inputs", {})
    for key, words in oou.items():
        if s == key or any(w == s or w in s for w in words):
            return InputSpec("out_of_universe", key, raw=raw)
    return InputSpec("unknown", None, raw=raw)
