"""Verification context: the panel, windows, executor, thresholds and caches shared by all verifiers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from configs import codebook, study, thresholds
from data.labels import daily_returns, forward_returns
from data.panel import Panel
from dsl import Node, canonical_hash, parse
from executors import E2Executor


@dataclass
class VerificationContext:
    panel: Panel
    windows: dict = field(default_factory=dict)        # name -> (start, end)
    thr: dict = field(default_factory=thresholds)
    cb: dict = field(default_factory=codebook)
    executor: object = field(default_factory=E2Executor)
    external_factors: pd.DataFrame | None = None        # e.g. Ken French daily table (decimal returns)
    seed: int = 20261006
    fast: bool = False                                  # smaller bootstrap/nudge sizes (tests, smoke runs)

    def __post_init__(self) -> None:
        if not self.windows:
            sp = study()["splits"]["comparability"]
            self.windows = {k: tuple(sp[k]) for k in ("train", "valid", "test") if k in sp}
        self._signals: dict[str, np.ndarray] = {}
        self._fwd: dict[tuple, np.ndarray] = {}
        self._refs = None
        self._factors = None

    # ------------------------------------------------------------------ windows
    def rows(self, window: str) -> np.ndarray:
        if window == "all":
            return np.ones(self.panel.T, dtype=bool)
        start, end = self.windows[window]
        return self.panel.date_mask(start, end)

    def has_window(self, window: str) -> bool:
        return window in self.windows and self.rows(window).sum() > 30

    # ------------------------------------------------------------------ signals
    def signal(self, node: Node | str) -> np.ndarray:
        if isinstance(node, str):
            node = parse(node)
        key = canonical_hash(node) + "|" + repr(node)
        if key not in self._signals:
            self._signals[key] = self.executor.evaluate(node, self.panel)
        return self._signals[key]

    def put_signal(self, key: str, arr: np.ndarray) -> None:
        self._signals[key] = arr

    def fwd(self, h: int = 1, entry: str = "close_t") -> np.ndarray:
        k = (h, entry)
        if k not in self._fwd:
            self._fwd[k] = forward_returns(self.panel, h, entry)
        return self._fwd[k]

    def daily_returns(self) -> np.ndarray:
        if ("ret", 0) not in self._fwd:
            self._fwd[("ret", 0)] = daily_returns(self.panel)
        return self._fwd[("ret", 0)]

    def market_returns(self) -> np.ndarray:
        """Equal-weighted member return per day."""
        import warnings

        r = np.where(self.panel.member, self.daily_returns(), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(r, axis=1)

    # ------------------------------------------------------------------ lazily built libraries
    @property
    def references(self):
        if self._refs is None:
            from .references import ReferenceLibrary

            self._refs = ReferenceLibrary(self)
        return self._refs

    @property
    def factors(self):
        if self._factors is None:
            from .factors import FactorSet

            self._factors = FactorSet(self)
        return self._factors

    # ------------------------------------------------------------------ sizes
    def n_boot(self) -> int:
        return 200 if self.fast else int(self.thr["behavioral"]["bootstrap"]["n_boot"])

    def n_contexts(self) -> int:
        return 120 if self.fast else int(self.thr["nudge"]["n_contexts"])

    # ------------------------------------------------------------------ scopes
    def scope_mask(self, scope: str | None) -> np.ndarray | None:
        """Stock-day mask for conditional claims ("in small caps"), from the codebook scopes."""
        if not scope:
            return None
        scopes = self.cb.get("scopes", {})
        key = scope.strip().lower()
        spec = scopes.get(key)
        if spec is None:
            for k, v in scopes.items():
                if k in key or key in k:
                    spec = v
                    break
        if spec is None:
            return None
        sig = self.references.signal(spec["signal"])
        pct = pd.DataFrame(np.where(self.panel.member, sig, np.nan)).rank(axis=1, pct=True).to_numpy()
        if spec["tercile"] == "bottom":
            return pct <= 1 / 3
        return pct > 2 / 3
