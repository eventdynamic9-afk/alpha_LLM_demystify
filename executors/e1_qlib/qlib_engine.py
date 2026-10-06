"""Optional adapter that evaluates formulas with the real Qlib expression engine (§6.4 E1).

Every subtree that contains no cross-sectional operator is sent to ``qlib.data.D.features`` as a
strict Qlib expression; cross-sectional operators (and any operator above them) are applied by the
pandas layer of :class:`executors.e1_qlib.engine.E1Executor`.  Requires ``pyqlib`` installed and
``qlib.init(provider_uri=...)`` called on the same data the panel was built from.  Note that Qlib's
native rolling operators use ``min_periods=1``; agreement checks therefore compare after masking
each window's warm-up (``E1Executor(qlib_native=True)`` reproduces that behaviour without Qlib).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dsl.ast import Node, walk
from dsl.operators import OPS, XS
from dsl.serialize import to_qlib

from .engine import E1Executor


def qlib_available() -> bool:
    try:
        import qlib  # noqa: F401
        from qlib.data import D  # noqa: F401
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


class QlibExecutor(E1Executor):  # pragma: no cover - requires pyqlib and Qlib data
    name = "E1-qlib"

    def __init__(self, instruments: list[str], start: str, end: str):
        super().__init__(qlib_native=True)
        if not qlib_available():
            raise RuntimeError("pyqlib is not installed; use E1Executor (pandas mirror) instead")
        self.instruments = instruments
        self.start, self.end = start, end

    def _eval(self, n: Node) -> pd.DataFrame:
        if n in self._cache:
            return self._cache[n]
        has_xs = any(not m.is_leaf and OPS[m.op].kind == XS for m in walk(n))
        if not has_xs and not n.is_const:
            from qlib.data import D

            expr = to_qlib(n, strict=True)
            raw = D.features(self.instruments, [expr], start_time=self.start, end_time=self.end)
            wide = raw.iloc[:, 0].unstack(level=0)
            wide = wide.reindex(index=pd.DatetimeIndex(self._panel.dates),
                                columns=[i.upper() for i in self._panel.instruments])
            df = self._frame(wide.to_numpy(dtype=np.float64))
            self._cache[n] = df
            return df
        return super()._eval(n)
