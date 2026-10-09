"""SQLite-backed hyperparameter-trial store.

Each finished trial is committed before the next starts. A restarted
``--stage tune`` skips rows that are already in the database, so a crash
does not redo completed (architecture × target × trial_index) work.

This is a persistence layer over the existing seeded random search — it
does not change the sampler (not Optuna TPE). Study names are
``{kind}_{target}`` (e.g. ``lstm_temperature``).
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return float(obj)
    return obj


class TrialStore:
    """Append-only SQLite study. Safe to reopen after a kill/OOM."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=60000")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trials (
                study TEXT NOT NULL,
                trial_index INTEGER NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (study, trial_index)
            )
            """
        )
        self.conn.commit()

    def has(self, study: str, trial_index: int) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM trials WHERE study=? AND trial_index=?",
            (study, int(trial_index)),
        ).fetchone()
        return row is not None

    def get(self, study: str, trial_index: int) -> dict | None:
        row = self.conn.execute(
            "SELECT payload FROM trials WHERE study=? AND trial_index=?",
            (study, int(trial_index)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def rows(self, study: str) -> list[dict]:
        cur = self.conn.execute(
            "SELECT payload FROM trials WHERE study=? ORDER BY trial_index",
            (study,),
        )
        return [json.loads(r[0]) for r in cur.fetchall()]

    def put(self, study: str, trial_index: int, payload: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trials (study, trial_index, payload, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (study, int(trial_index), json.dumps(_jsonable(payload)), now),
        )
        self.conn.commit()


def self_check(path: str) -> None:
    """Minimal before/after: a written trial is visible after reopen."""
    s = TrialStore(path)
    s.put("lstm_temperature", 0, {"trial": 0, "rmse": 1.25})
    s.conn.close()
    s2 = TrialStore(path)
    assert s2.has("lstm_temperature", 0)
    assert not s2.has("lstm_temperature", 1)
    assert abs(s2.get("lstm_temperature", 0)["rmse"] - 1.25) < 1e-9
    s2.conn.close()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        self_check(os.path.join(td, "study.sqlite"))
    print("trial_store ok")
