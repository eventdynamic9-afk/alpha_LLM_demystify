"""Shared download helpers for rebuild scripts. Every downloaded file is hashed into MANIFEST.csv."""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

from configs import REPO_ROOT

from ..manifest import add_entry

RAW = REPO_ROOT / "data" / "raw"
PROCESSED = REPO_ROOT / "data" / "processed"


def download(url: str, dest: str | Path, source: str, license: str = "", notes: str = "",
             retries: int = 4, headers: dict | None = None) -> Path:
    """Download with exponential back-off (2, 4, 8, 16 s); honours HTTPS_PROXY and the CA bundle."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "rationale-fidelity/2.0", **(headers or {})})
    delay = 2
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            break
        except Exception:
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
    add_entry(dest, source, url, license, notes)
    return dest


def env_key(name: str) -> str | None:
    v = os.environ.get(name)
    return v if v else None
