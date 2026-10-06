"""Call logging and response cache (§4.3).

Every call is logged as JSONL: request (system + user messages, tool schema), response (text, tool
calls, reasoning tokens if exposed), parameters, timestamps, latency, token counts, model fingerprint
and the SHA-256 of the prompt template.  Responses are cached keyed by (model, prompt hash, sample
index, temperature); a cached response is never regenerated silently.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from pools.records import now_iso


def prompt_hash(messages: list[dict], tools: list | None = None) -> str:
    return hashlib.sha256(json.dumps({"m": messages, "t": tools}, sort_keys=True).encode()).hexdigest()


class CallLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, **record) -> None:
        record.setdefault("logged_at", now_iso())
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")


class ResponseCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, phash TEXT, "
                         "sample INTEGER, temperature REAL, response TEXT, created_at TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(model: str, phash: str, sample: int, temperature: float) -> str:
        return f"{model}|{phash}|{sample}|{temperature:.3f}"

    def get(self, model: str, phash: str, sample: int, temperature: float) -> dict | None:
        k = self.key(model, phash, sample, temperature)
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, phash: str, sample: int, temperature: float, response: dict) -> None:
        k = self.key(model, phash, sample, temperature)
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO cache VALUES (?,?,?,?,?,?,?)",
                             (k, model, phash, sample, temperature, json.dumps(response, default=str), now_iso()))
            self._db.commit()
