"""Data-field registry: names in each notation, units, value domains and type-compatible swaps."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    name: str
    description: str
    kind: str                      # price | volume | amount
    unit: tuple                    # sorted ((dimension, exponent), ...)
    lo: float                      # value domain lower bound
    hi: float
    lo_open: bool                  # True -> strictly greater than lo
    swaps: tuple = field(default=())   # SA-field type-compatible swaps (§7.2)

    @property
    def qlib(self) -> str:
        return f"${self.name}"

    @property
    def alpha101(self) -> str:
        return self.name

    @property
    def gtja(self) -> str:
        return self.name.upper()


_P = (("price", 1.0),)
_V = (("shares", 1.0),)
_A = (("price", 1.0), ("shares", 1.0))

FIELDS: dict[str, FieldSpec] = {
    "open": FieldSpec("open", "daily open price, split/dividend adjusted", "price", _P, 0.0, math.inf, True, ("close",)),
    "high": FieldSpec("high", "daily high price, split/dividend adjusted", "price", _P, 0.0, math.inf, True, ("low",)),
    "low": FieldSpec("low", "daily low price, split/dividend adjusted", "price", _P, 0.0, math.inf, True, ("high",)),
    "close": FieldSpec("close", "daily close price, split/dividend adjusted", "price", _P, 0.0, math.inf, True, ("open", "vwap")),
    "vwap": FieldSpec("vwap", "daily volume-weighted average price, adjusted", "price", _P, 0.0, math.inf, True, ("close",)),
    "volume": FieldSpec("volume", "daily traded volume (shares), split adjusted", "volume", _V, 0.0, math.inf, False, ("amount",)),
    "amount": FieldSpec("amount", "daily traded value (currency)", "amount", _A, 0.0, math.inf, False, ("volume",)),
}

PRICE_FIELDS = tuple(n for n, f in FIELDS.items() if f.kind == "price")


def field_glossary(names, market: str = "CN", notation: str = "qlib", legend: dict | None = None) -> str:
    """One line per field, in the order given (callers randomize the order, §8.4)."""
    lines = []
    for n in names:
        spec = FIELDS[n]
        desc = spec.description
        if n == "vwap" and market == "US":
            desc = "proxy (high+low+close)/3 because daily VWAP is unavailable for this market"
        if legend is not None:
            label = legend[n]
        elif notation == "qlib":
            label = spec.qlib
        elif notation == "gtja":
            label = spec.gtja
        else:
            label = spec.alpha101
        lines.append(f"{label}: {desc}")
    return "; ".join(lines)
