"""Shared cache and work queue between the training workers and the advisor.

Workers only ever read this, and only on the fast path. Every method swallows
its own errors and degrades to "no advice": a broken cache must never raise into
the training step loop, because the whole feature is optional.
"""

import sqlite3
import time

from gametext import normalise

NO_ADVICE = 8  # one past the 8 action indices

# A transient failure, typically a lock timeout under load, must not disable
# advice for the rest of the run. Back off, then try to reconnect. A
# genuinely broken file simply fails again and backs off again, so the cost
# of a permanent fault is one reconnect attempt per interval.
RETRY_SECONDS = 60.0

# Connection busy timeout. Exposed as a module constant so the concurrency
# test can shorten it: at the production value a reader simply waits out a
# rollback-journal lock, which makes the test pass even without WAL and so
# proves nothing.
BUSY_TIMEOUT = 5.0


class AdviceCache:
    def __init__(self, path: str = "advice.db", readonly: bool = False):
        self.path = path
        self.readonly = readonly
        self._conn: sqlite3.Connection | None = None
        self._broken_until = 0.0
        self._connect()

    def _connect(self) -> None:
        try:
            conn = sqlite3.connect(self.path, timeout=BUSY_TIMEOUT, check_same_thread=False)
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
            self._broken_until = time.monotonic() + RETRY_SECONDS
            self._conn = None

    def _mark_broken(self) -> None:
        self._broken_until = time.monotonic() + RETRY_SECONDS
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    def _usable(self) -> bool:
        """True if the connection is live, reconnecting once the backoff ends."""
        if self._conn is not None:
            return True
        if time.monotonic() < self._broken_until:
            return False
        self._connect()
        return self._conn is not None

    def lookup(self, text: str) -> int:
        if not self._usable() or not text:
            return NO_ADVICE
        try:
            row = self._conn.execute(
                "SELECT action FROM advice WHERE text = ?", (normalise(text),)
            ).fetchone()
        except Exception:
            self._mark_broken()
            return NO_ADVICE
        return row[0] if row else NO_ADVICE

    def enqueue(self, text: str) -> None:
        if not self._usable() or self.readonly or not text:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO queue (text) VALUES (?)", (normalise(text),)
            )
            self._conn.commit()
        except Exception:
            self._mark_broken()

    def pending(self, limit: int = 32) -> list[str]:
        if not self._usable():
            return []
        try:
            rows = self._conn.execute(
                "SELECT text FROM queue LIMIT ?", (limit,)
            ).fetchall()
        except Exception:
            self._mark_broken()
            return []
        return [r[0] for r in rows]

    def put(self, text: str, action: int) -> None:
        if not self._usable() or not text:
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
            self._mark_broken()
