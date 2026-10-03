"""Shared cache and work queue between the training workers and the advisor.

Workers only ever read this, and only on the fast path. Every method swallows
its own errors and degrades to "no advice": a broken cache must never raise into
the training step loop, because the whole feature is optional.
"""

import sqlite3

from gametext import normalise

NO_ADVICE = 8  # one past the 8 action indices


class AdviceCache:
    def __init__(self, path: str = "advice.db", readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._conn: sqlite3.Connection | None = None
        self._broken = False
        self._connect()

    def _connect(self) -> None:
        try:
            conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            # WAL lets eight workers read while the advisor writes. The default
            # rollback journal takes an exclusive lock and raises
            # "database is locked" under this access pattern.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS advice ("
                "  text TEXT PRIMARY KEY, action INTEGER NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS queue (text TEXT PRIMARY KEY)"
            )
            conn.commit()
            self._conn = conn
        except Exception:
            # Corrupt file, unwritable path, missing directory: all the same
            # outcome. The feature switches itself off.
            self._broken = True
            self._conn = None

    def lookup(self, text: str) -> int:
        if self._broken or self._conn is None or not text:
            return NO_ADVICE
        try:
            row = self._conn.execute(
                "SELECT action FROM advice WHERE text = ?", (normalise(text),)
            ).fetchone()
        except Exception:
            self._broken = True
            return NO_ADVICE
        return row[0] if row else NO_ADVICE

    def enqueue(self, text: str) -> None:
        if self._broken or self._conn is None or self.readonly or not text:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO queue (text) VALUES (?)", (normalise(text),)
            )
            self._conn.commit()
        except Exception:
            self._broken = True

    def pending(self, limit: int = 32) -> list[str]:
        if self._broken or self._conn is None:
            return []
        try:
            rows = self._conn.execute(
                "SELECT text FROM queue LIMIT ?", (limit,)
            ).fetchall()
        except Exception:
            self._broken = True
            return []
        return [r[0] for r in rows]

    def put(self, text: str, action: int) -> None:
        if self._broken or self._conn is None or not text:
            return
        key = normalise(text)
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO advice (text, action) VALUES (?, ?)",
                (key, int(action)),
            )
            self._conn.execute("DELETE FROM queue WHERE text = ?", (key,))
            self._conn.commit()
        except Exception:
            self._broken = True
