"""Cache SQLite pour les appels LLM.

Evite de refaire les memes appels API sur des entrees identiques.
La cle de cache est un hash SHA-256 de (model, temperature, seed, messages).
"""

import hashlib
import json
import os
import sqlite3
from pathlib import Path

_DEFAULT_DB = Path("data/cache/llm_cache.db")

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    key      TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class LLMCache:
    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            db_path.parent.chmod(0o700)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(_INIT_SQL)
        self._conn.commit()
        if os.name != "nt" and db_path.exists():
            db_path.chmod(0o600)

    def _make_key(self, model: str, temperature: float, seed: int | None, messages: list[dict]) -> str:
        payload = json.dumps(
            {"model": model, "temperature": temperature, "seed": seed, "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, temperature: float, seed: int | None, messages: list[dict]) -> str | None:
        key = self._make_key(model, temperature, seed, messages)
        row = self._conn.execute("SELECT response FROM llm_cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, model: str, temperature: float, seed: int | None, messages: list[dict], response: str) -> None:
        key = self._make_key(model, temperature, seed, messages)
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, response) VALUES (?, ?)",
            (key, response),
        )
        self._conn.commit()

    def clear(self) -> None:
        """Delete cached responses, for retention and incident-response flows."""
        self._conn.execute("DELETE FROM llm_cache")
        self._conn.commit()
