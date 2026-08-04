"""Step-1 tests: deterministic environment, state oracle, structural invariants.

No agent, no tools, no policy here, only the environment contract:
byte-identical fresh seeds, complete deterministic snapshots, append-only
events, and a self-consistent seed.
"""

import hashlib
import json
import sqlite3

import pytest

from telecom_aut.environment import SIM_EPOCH, TelecomEnv

EXPECTED_TABLES = [
    "subscribers",
    "plans",
    "orders",
    "technicians",
    "availability_slots",
    "appointments",
    "invoices",
    "policy_documents",
    "events",
]


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def env(tmp_path):
    e = TelecomEnv.fresh(tmp_path / "env.db")
    yield e
    e.close()


# -- determinism: two fresh builds must be byte-identical -------------------

def test_two_fresh_databases_are_byte_identical(tmp_path):
    a, b = tmp_path / "a.db", tmp_path / "b.db"
    TelecomEnv.fresh(a).close()
    TelecomEnv.fresh(b).close()
    assert sha256(a) == sha256(b)


def test_snapshots_of_fresh_seeds_are_identical(tmp_path):
    ea = TelecomEnv.fresh(tmp_path / "a.db")
    eb = TelecomEnv.fresh(tmp_path / "b.db")
    try:
        assert ea.snapshot() == eb.snapshot()
    finally:
        ea.close()
        eb.close()


def test_reset_after_mutation_restores_pristine_bytes_and_state(tmp_path):
    pristine = tmp_path / "pristine.db"
    TelecomEnv.fresh(pristine).close()

    e = TelecomEnv.fresh(tmp_path / "mutated.db")
    baseline = e.snapshot()
    e.conn.execute(
        "UPDATE subscribers SET full_name = 'Mallory' WHERE id = 'SUB-0001'"
    )
    e.conn.execute(
        "INSERT INTO events (sim_time, actor, operation, entity_type, entity_id)"
        " VALUES (?, 'agent', 'update_subscriber', 'subscribers', 'SUB-0001')",
        (e.tick(),),
    )
    e.conn.commit()
    assert e.snapshot() != baseline

    e.reset_to_seed()
    assert e.snapshot() == baseline
    assert e.sim_now == SIM_EPOCH  # clock reset too
    e.close()
    assert sha256(tmp_path / "mutated.db") == sha256(pristine)


# -- snapshot contract ------------------------------------------------------

def test_snapshot_covers_all_tables_in_fixed_order(env):
    snap = env.snapshot()
    assert snap["schema_version"] == 2
    assert list(snap["tables"].keys()) == EXPECTED_TABLES


def test_snapshot_rows_are_pk_ordered_and_json_safe(env):
    snap = env.snapshot()
    for table, rows in snap["tables"].items():
        pk = "seq" if table == "events" else "id"
        keys = [r[pk] for r in rows]
        assert keys == sorted(keys), f"{table} not ordered by {pk}"
    # JSON round-trip must be lossless: only int/str/None values.
    assert json.loads(json.dumps(snap)) == snap


def test_repeated_snapshots_without_writes_are_equal(env):
    assert env.snapshot() == env.snapshot()


def test_seed_starts_with_no_events_and_full_policy_corpus(env):
    snap = env.snapshot()["tables"]
    assert snap["events"] == []  # every final-state event is agent-caused
    assert len(snap["policy_documents"]) == 12  # the full retrieval corpus


# -- seed self-consistency --------------------------------------------------

def test_seed_passes_foreign_key_check(env):
    assert env.conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_booked_slots_match_non_cancelled_appointments_exactly(env):
    snap = env.snapshot()["tables"]
    booked = {s["id"] for s in snap["availability_slots"] if s["status"] == "booked"}
    referenced = [
        a["slot_id"] for a in snap["appointments"]
        if a["status"] in ("pending", "completed")
    ]
    assert len(referenced) == len(set(referenced)), "a slot is double-booked"
    assert set(referenced) == booked


def test_seed_has_free_slots_for_rescheduling(env):
    snap = env.snapshot()["tables"]
    free = [s for s in snap["availability_slots"] if s["status"] == "available"]
    assert len(free) >= 10  # rescheduling must always have real alternatives


# -- structural enforcement (allowed), policy enforcement (absent) ----------

def test_events_table_is_append_only(env):
    env.conn.execute(
        "INSERT INTO events (sim_time, actor, operation, entity_type)"
        " VALUES (?, 'agent', 'noop', 'none')",
        (env.tick(),),
    )
    with pytest.raises(sqlite3.IntegrityError):
        env.conn.execute("UPDATE events SET actor = 'tampered' WHERE seq = 1")
    with pytest.raises(sqlite3.IntegrityError):
        env.conn.execute("DELETE FROM events WHERE seq = 1")


def test_schema_rejects_structurally_invalid_rows(env):
    with pytest.raises(sqlite3.IntegrityError):  # unknown FK target
        env.conn.execute(
            "INSERT INTO orders (id, subscriber_id, plan_id, state, address,"
            " region, created_at) VALUES ('ORD-9999', 'SUB-4242', 10,"
            " 'active', 'x', 'north', '2026-07-01T00:00:00Z')"
        )
    with pytest.raises(sqlite3.IntegrityError):  # enum violation
        env.conn.execute(
            "UPDATE appointments SET status = 'exploded' WHERE id = 'APT-0400'"
        )


def test_database_happily_stores_policy_violating_state(env):
    """The DB layer must not enforce policy.

    Cancelling Bruno's order (unpaid balance, no confirmation) violates
    policy but must succeed at the storage layer: policy is the agent's job.
    """
    env.conn.execute(
        "UPDATE orders SET state = 'cancelled', cancelled_at = ?"
        " WHERE id = 'ORD-0101'",
        (env.tick(),),
    )
    env.conn.commit()
    row = env.conn.execute(
        "SELECT state FROM orders WHERE id = 'ORD-0101'"
    ).fetchone()
    assert row["state"] == "cancelled"


# -- simulation clock -------------------------------------------------------

def test_sim_clock_is_deterministic(env):
    assert env.sim_now == "2026-07-28T09:00:00Z"
    assert env.tick() == "2026-07-28T09:00:01Z"
    assert env.tick() == "2026-07-28T09:00:02Z"
    env.reset_to_seed()
    assert env.sim_now == "2026-07-28T09:00:00Z"
