-- Telecom agent-under-test environment schema (schema_version = 2).
--
-- Domain model aligned with the pilot-fiber taxonomy:
--   subscribers -> orders (1..n, carry plan + address) -> appointments.
--
-- Design rules:
--   * The schema enforces STRUCTURE only (types, FKs, enum CHECKs, id
--     formats, append-only events). It must never enforce POLICY
--     (eligibility, confirmation, ordering). Policy checking is the
--     agent's job, and policy violations must be physically executable.
--   * Entity ids are prefixed hex strings (SUB-/ORD-/APT-/SLT-/INV- + 4 hex
--     chars, zero-padded) so a bare id is self-describing and a wrong-entity
--     id is structurally detectable.
--   * All timestamps are ISO-8601 UTC strings ("2026-07-28T09:00:00Z"),
--     so lexicographic order == chronological order.
--   * Money is integer cents. No floats anywhere.

PRAGMA user_version = 2;

CREATE TABLE subscribers (
    id         TEXT PRIMARY KEY CHECK (id GLOB 'SUB-[0-9A-F][0-9A-F][0-9A-F][0-9A-F]'),
    full_name  TEXT NOT NULL,
    email      TEXT NOT NULL UNIQUE,
    state      TEXT NOT NULL CHECK (state IN ('active', 'cancelled')),
    created_at TEXT NOT NULL
);

CREATE TABLE plans (
    id                  INTEGER PRIMARY KEY,
    code                TEXT NOT NULL UNIQUE,       -- tools reference plans by code
    name                TEXT NOT NULL,
    tier                INTEGER NOT NULL,           -- 1 = lowest. Policy docs refer to tiers
    monthly_price_cents INTEGER NOT NULL,
    speed_mbps          INTEGER NOT NULL
);

CREATE TABLE orders (
    id            TEXT PRIMARY KEY CHECK (id GLOB 'ORD-[0-9A-F][0-9A-F][0-9A-F][0-9A-F]'),
    subscriber_id TEXT NOT NULL REFERENCES subscribers(id),
    plan_id       INTEGER NOT NULL REFERENCES plans(id),
    state         TEXT NOT NULL
                  CHECK (state IN ('active', 'pending_installation', 'cancelled')),
    address       TEXT NOT NULL,
    region        TEXT NOT NULL,                    -- service region of the address
    created_at    TEXT NOT NULL,
    cancelled_at  TEXT                              -- NULL unless state = 'cancelled'
);

CREATE TABLE technicians (
    id        INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    region    TEXT NOT NULL
);

-- Slots have a real lifecycle: 'available' <-> 'booked'. Rescheduling requires
-- releasing the old slot and booking the new one, not a single UPDATE. An
-- appointment's date is its slot's start time (exposed by the read tools).
CREATE TABLE availability_slots (
    id            TEXT PRIMARY KEY CHECK (id GLOB 'SLT-[0-9A-F][0-9A-F][0-9A-F][0-9A-F]'),
    technician_id INTEGER NOT NULL REFERENCES technicians(id),
    starts_at     TEXT NOT NULL,
    ends_at       TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('available', 'booked')),
    UNIQUE (technician_id, starts_at)
);

CREATE TABLE appointments (
    id         TEXT PRIMARY KEY CHECK (id GLOB 'APT-[0-9A-F][0-9A-F][0-9A-F][0-9A-F]'),
    order_id   TEXT NOT NULL REFERENCES orders(id),
    slot_id    TEXT REFERENCES availability_slots(id),  -- NULL once released/cancelled
    kind       TEXT NOT NULL CHECK (kind IN ('installation', 'maintenance')),
    -- POLICY (not schema): maintenance visits belong to active orders,
    -- installation visits to orders pending installation.
    status     TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'cancelled')),
    notes      TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE invoices (
    id           TEXT PRIMARY KEY CHECK (id GLOB 'INV-[0-9A-F][0-9A-F][0-9A-F][0-9A-F]'),
    order_id     TEXT NOT NULL REFERENCES orders(id),
    amount_cents INTEGER NOT NULL,
    issued_on    TEXT NOT NULL,
    due_on       TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('paid', 'unpaid'))
    -- "overdue" is derived: status = 'unpaid' AND due_on < sim_now
);

-- Retrieval corpus.
CREATE TABLE policy_documents (
    id       INTEGER PRIMARY KEY,
    slug     TEXT NOT NULL UNIQUE,
    title    TEXT NOT NULL,
    category TEXT NOT NULL,
    body     TEXT NOT NULL
);

-- Append-only audit log of every write, populated by the pseudo-API layer.
-- Empty at seed time so every event in a final state is agent-caused.
CREATE TABLE events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    sim_time    TEXT NOT NULL,
    actor       TEXT NOT NULL,
    operation   TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT,
    payload     TEXT NOT NULL DEFAULT '{}'          -- JSON
);

-- Append-only is structural, so it is enforced here (unlike policy).
CREATE TRIGGER events_no_update BEFORE UPDATE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;

CREATE TRIGGER events_no_delete BEFORE DELETE ON events
BEGIN
    SELECT RAISE(ABORT, 'events is append-only');
END;
