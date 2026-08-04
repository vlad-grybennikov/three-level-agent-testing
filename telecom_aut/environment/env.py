"""TelecomEnv: the deterministic state oracle for the agent under test.

Responsibilities:
  * hold all agent-visible state in one SQLite database,
  * `reset_to_seed()`: rebuild the database from scratch (two fresh
    builds are byte-identical),
  * `snapshot()`: cheap, complete, deterministically ordered dump of the
    logical state, safe to call between any two tool calls,
  * a deterministic simulation clock for timestamps written at runtime
    (no wall-clock anywhere in environment state).

This module contains no policy and no scoring. The pseudo-API tool layer
mutates the database through `conn` and stamps writes with `tick()`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .seed import SIM_EPOCH, seed_rows

SCHEMA_VERSION = 2
_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Fixed dump order. Primary key per table gives deterministic row order
# (prefixed ids are fixed-width, so lexicographic order is stable).
_TABLE_PKS: dict[str, str] = {
    "subscribers": "id",
    "plans": "id",
    "orders": "id",
    "technicians": "id",
    "availability_slots": "id",
    "appointments": "id",
    "invoices": "id",
    "policy_documents": "id",
    "events": "seq",
}

_EPOCH_DT = datetime.strptime(SIM_EPOCH, "%Y-%m-%dT%H:%M:%SZ").replace(
    tzinfo=timezone.utc
)


class TelecomEnv:
    """One episode's environment: a seeded SQLite DB plus a simulation clock."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._clock_offset = 0  # seconds since SIM_EPOCH
        self._conn: sqlite3.Connection | None = None
        self._connect()

    @classmethod
    def fresh(cls, db_path: str | Path = ":memory:") -> "TelecomEnv":
        """Construct and seed in one step."""
        env = cls(db_path)
        env.reset_to_seed()
        return env

    # -- connection ---------------------------------------------------------

    @property
    def conn(self) -> sqlite3.Connection:
        assert self._conn is not None, "environment is closed"
        return self._conn

    def _connect(self) -> None:
        # check_same_thread=False: the chat demo serves from worker threads.
        # Writes stay serialized (episode lock + SQLite serialized mode).
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Fix the page size before any content exists so the file layout does
        # not depend on the local SQLite compile-time default.
        conn.execute("PRAGMA page_size = 4096")
        conn.execute("PRAGMA foreign_keys = ON")
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- seeding ------------------------------------------------------------

    def reset_to_seed(self) -> None:
        """Rebuild the database from scratch and reset the simulation clock.

        The file (if any) is deleted and recreated so that the byte content
        depends only on the schema and seed data, never on prior mutations.
        """
        self.close()
        if self.db_path != ":memory:":
            for suffix in ("", "-journal", "-wal", "-shm"):
                Path(self.db_path + suffix).unlink(missing_ok=True)
        self._connect()

        conn = self.conn
        conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        for table, rows in seed_rows().items():
            for row in rows:
                cols = ", ".join(row)
                params = ", ".join(f":{c}" for c in row)
                conn.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({params})", row
                )
        conn.commit()
        self._clock_offset = 0

    # -- state oracle -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Complete logical state, deterministically ordered, JSON-safe.

        Row dicts preserve schema column order, tables appear in fixed
        order, and rows are sorted by primary key. Values are int / str /
        None only.
        """
        tables: dict[str, list[dict[str, Any]]] = {}
        for table, pk in _TABLE_PKS.items():
            cur = self.conn.execute(f"SELECT * FROM {table} ORDER BY {pk}")
            cols = [d[0] for d in cur.description]
            tables[table] = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"schema_version": SCHEMA_VERSION, "tables": tables}

    # -- simulation clock ---------------------------------------------------

    @property
    def sim_now(self) -> str:
        """Current simulated time as an ISO-8601 UTC string."""
        dt = _EPOCH_DT + timedelta(seconds=self._clock_offset)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def tick(self) -> str:
        """Advance the clock by one second and return the new time.

        Every runtime write stamps itself via tick(), so timestamps are
        unique, ordered, and identical across runs with the same call
        sequence.
        """
        self._clock_offset += 1
        return self.sim_now
