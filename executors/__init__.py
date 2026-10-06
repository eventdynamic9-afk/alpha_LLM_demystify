"""Two independent executors for the typed DSL (§6.4) plus agreement and causality checks."""
from .e1_qlib import E1Executor
from .e2_numpy import E2Executor


def get_executor(name: str = "e2"):
    name = name.lower()
    if name in ("e2", "numpy", "e2-numpy"):
        return E2Executor()
    if name in ("e1", "pandas", "e1-qlib-semantics"):
        return E1Executor()
    if name in ("e1-native", "qlib-native"):
        return E1Executor(qlib_native=True)
    raise ValueError(f"unknown executor {name!r}")


def evaluate(node, panel, engine: str = "e2", mask_members: bool = True):
    return get_executor(engine).evaluate(node, panel, mask_members=mask_members)


__all__ = ["E1Executor", "E2Executor", "get_executor", "evaluate"]
