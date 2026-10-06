"""Loaders for the frozen configuration files (thresholds, study design, codebook, roster, prompts)."""
from __future__ import annotations

import functools
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent
PROMPT_DIR = CONFIG_DIR / "prompts"


def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    p = Path(name_or_path)
    if not p.is_absolute() and not p.exists():
        p = CONFIG_DIR / p
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=None)
def thresholds() -> dict[str, Any]:
    return load_yaml("thresholds.yaml")


@functools.lru_cache(maxsize=None)
def study() -> dict[str, Any]:
    return load_yaml("study.yaml")


@functools.lru_cache(maxsize=None)
def codebook() -> dict[str, Any]:
    return load_yaml("codebook.yaml")


def models(path: str | Path = "models.yaml") -> dict[str, Any]:
    return load_yaml(path)


def prompt(name: str) -> str:
    """Return a prompt template verbatim (Appendix A/B); trailing newline stripped."""
    fname = name if name.endswith((".txt", ".yaml")) else f"{name}.txt"
    return (PROMPT_DIR / fname).read_text(encoding="utf-8").rstrip("\n")


_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)(:[^{}]*)?\}")


def fill(template: str, **values: Any) -> str:
    """Substitute ``{name}`` / ``{name:fmt}`` placeholders, leaving literal JSON braces untouched.

    Unknown placeholders raise ``KeyError`` so a template/field mismatch can never pass silently.
    """

    def repl(m: re.Match[str]) -> str:
        key, fmt = m.group(1), m.group(2)
        if key not in values:
            raise KeyError(f"template placeholder {{{key}}} not supplied")
        v = values[key]
        return format(v, fmt[1:]) if fmt else str(v)

    return _PLACEHOLDER.sub(repl, template)


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thresholds_hash() -> str:
    """SHA-256 of the thresholds file, recorded in the pre-registration (Appendix E.3)."""
    return file_sha256(CONFIG_DIR / "thresholds.yaml")
