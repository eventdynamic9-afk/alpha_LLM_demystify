"""data/MANIFEST.csv: every downloaded file hashed with source URL and access date (§5.9)."""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from configs import REPO_ROOT, file_sha256

MANIFEST = REPO_ROOT / "data" / "MANIFEST.csv"
COLUMNS = ["path", "source", "url", "access_date", "sha256", "size_bytes", "license", "notes"]


def add_entry(path: str | Path, source: str, url: str, license: str = "", notes: str = "",
              manifest: str | Path = MANIFEST) -> dict:
    path = Path(path)
    row = {"path": str(path.resolve().relative_to(REPO_ROOT)) if path.resolve().is_relative_to(REPO_ROOT)
           else str(path), "source": source, "url": url, "access_date": dt.date.today().isoformat(),
           "sha256": file_sha256(path), "size_bytes": path.stat().st_size, "license": license, "notes": notes}
    rows = read_manifest(manifest)
    rows = [r for r in rows if r["path"] != row["path"]] + [row]
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return row


def read_manifest(manifest: str | Path = MANIFEST) -> list[dict]:
    if not Path(manifest).exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verify_manifest(manifest: str | Path = MANIFEST) -> list[str]:
    bad = []
    for r in read_manifest(manifest):
        p = Path(r["path"])
        p = p if p.is_absolute() else REPO_ROOT / p
        if not p.exists() or file_sha256(p) != r["sha256"]:
            bad.append(r["path"])
    return bad
