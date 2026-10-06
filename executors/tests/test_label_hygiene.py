"""§6.3 layer 4: the expression engine can never see labels — no module under dsl/ or executors/
imports data.labels (static import scan)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            out.add(node.module or "")
            out.update(f"{node.module}.{a.name}" for a in node.names)
    return out


def test_engine_cannot_import_labels():
    offenders = []
    for pkg in ("dsl", "executors"):
        for py in (ROOT / pkg).rglob("*.py"):
            if "tests" in py.parts:
                continue
            imps = _imports(py)
            if any("labels" in i for i in imps):
                offenders.append(str(py))
    assert not offenders, offenders
