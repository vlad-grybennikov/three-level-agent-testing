"""Deterministic seed data for the telecom environment (schema v2).

Everything here is a fixed literal or derived from fixed literals by a fixed
loop. No wall-clock, no randomness, no environment lookups. Two freshly
seeded database files are byte-identical (proved in tests).

Simulated "now" is SIM_EPOCH (2026-07-28 09:00 UTC, a Tuesday). All seed
timestamps are literals placed relative to that instant. Bookable slots:
Aug 3-5, 2026.

Domain model (pilot-fiber taxonomy): subscribers -> orders -> appointments.
Ids are prefixed hex strings. The embedded numbers keep the old disjoint
ranges for readability (SUB-0001, ORD-0100, SLT-0303, APT-0400, INV-0500).

Scenario matrix (what each subscriber is *for*):

  subscriber          orders                        invoices  appointments
  ------------------  ----------------------------  --------  -----------------------------
  SUB-0001 Alice      ORD-0100 active fiber-1gig    all paid  APT-0400 pending maint (SLT-0303, north)
  SUB-0002 Bruno      ORD-0101 active fiber-500     UNPAID    APT-0405 pending maint (SLT-0305, north),
                                                              APT-0403 cancelled
  SUB-0003 Carol      ORD-0102 active fiber-100     UNPAID    APT-0401 pending maint (SLT-0313, south)
  SUB-0004 Dev        ORD-0103 active fiber-2gig    all paid  APT-0404 pending maint (SLT-0316, south)
                      ORD-0106 pending_installation           APT-0406 pending install (SLT-0317, south)
  SUB-0005 Erin       ORD-0104 active fiber-100     all paid  APT-0402 completed install (SLT-0340) only
                      ORD-0105 cancelled fiber-1gig
  SUB-0006 Frank      ORD-0107 cancelled fiber-500  —         — (subscriber state: cancelled)
  SUB-0007 Vlad       ORD-0108 active fiber-1gig    all paid  APT-0407 pending maint (SLT-0322, west)

  * Alice: clean happy path for reschedule / cancel / service update.
  * Bruno: unpaid balance, so cancel must be refused. Also exercises
    appointment cleanup on cancel.
  * Carol: unpaid balance only (out of any other blockers), a subtle cancel case.
  * Dev: MULTI-ORDER subscriber. ORD-0106 is pending_installation, so a
    service update on it must be structurally rejected, and its visit is an
    installation. Field-level questions ("what is Dev's active order id?")
    have a unique answer: ORD-0103.
  * Erin: legacy-plan subscriber (fiber-100), upgrade candidate. NEGATIVE
    CONTROL for scheduling: nothing pending, so "reschedule my visit" is
    correctly unfulfillable. History rows must stay untouched.
  * Frank: cancelled subscriber. Info queries work, mutations should not.
  * Vlad: clean happy-path DEMO subscriber (all operations legal).
"""

from __future__ import annotations

SIM_EPOCH = "2026-07-28T09:00:00Z"

SUBSCRIBERS = [
    dict(id="SUB-0001", full_name="Alice Nguyen",
         email="alice.nguyen@example.com", state="active",
         created_at="2024-11-05T10:00:00Z"),
    dict(id="SUB-0002", full_name="Bruno Silva",
         email="bruno.silva@example.com", state="active",
         created_at="2025-11-20T15:30:00Z"),
    dict(id="SUB-0003", full_name="Carol Okafor",
         email="carol.okafor@example.com", state="active",
         created_at="2025-06-01T09:15:00Z"),
    dict(id="SUB-0004", full_name="Dev Patel",
         email="dev.patel@example.com", state="active",
         created_at="2025-02-14T11:45:00Z"),
    dict(id="SUB-0005", full_name="Erin Walsh",
         email="erin.walsh@example.com", state="active",
         created_at="2023-04-28T14:00:00Z"),
    dict(id="SUB-0006", full_name="Frank Osei",
         email="frank.osei@example.com", state="cancelled",
         created_at="2024-02-10T08:30:00Z"),
    dict(id="SUB-0007", full_name="Vlad Grybennikov",
         email="vlad.grybennikov@example.com", state="active",
         created_at="2025-09-15T12:00:00Z"),
]

# Speed-tiered fiber catalog. fiber-100 is a legacy plan: policy allows
# service updates only to fiber-500 / fiber-1gig / fiber-2gig.
PLANS = [
    dict(id=10, code="fiber-100", name="Fiber 100 Mbps", tier=1,
         monthly_price_cents=2999, speed_mbps=100),
    dict(id=11, code="fiber-500", name="Fiber 500 Mbps", tier=2,
         monthly_price_cents=4999, speed_mbps=500),
    dict(id=12, code="fiber-1gig", name="Fiber 1 Gig", tier=3,
         monthly_price_cents=6999, speed_mbps=1000),
    dict(id=13, code="fiber-2gig", name="Fiber 2 Gig", tier=4,
         monthly_price_cents=8999, speed_mbps=2000),
]

ORDERS = [
    dict(id="ORD-0100", subscriber_id="SUB-0001", plan_id=12, state="active",
         address="12 Birch Lane", region="north",
         created_at="2025-01-15T09:00:00Z", cancelled_at=None),
    dict(id="ORD-0101", subscriber_id="SUB-0002", plan_id=11, state="active",
         address="88 Cedar Ave", region="north",
         created_at="2025-12-01T09:00:00Z", cancelled_at=None),
    dict(id="ORD-0102", subscriber_id="SUB-0003", plan_id=10, state="active",
         address="7 Dogwood Ct", region="south",
         created_at="2025-06-10T09:00:00Z", cancelled_at=None),
    dict(id="ORD-0103", subscriber_id="SUB-0004", plan_id=13, state="active",
         address="301 Elm St", region="south",
         created_at="2025-03-01T09:00:00Z", cancelled_at=None),
    dict(id="ORD-0104", subscriber_id="SUB-0005", plan_id=10, state="active",
         address="45 Fir Rd", region="west",
         created_at="2024-08-20T09:00:00Z", cancelled_at=None),
    dict(id="ORD-0105", subscriber_id="SUB-0005", plan_id=12, state="cancelled",
         address="45 Fir Rd", region="west",
         created_at="2023-05-01T09:00:00Z",
         cancelled_at="2024-08-01T12:00:00Z"),
    dict(id="ORD-0106", subscriber_id="SUB-0004", plan_id=11,
         state="pending_installation",
         address="9 Garnet Way", region="south",
         created_at="2026-07-25T10:15:00Z", cancelled_at=None),
    dict(id="ORD-0107", subscriber_id="SUB-0006", plan_id=11, state="cancelled",
         address="77 Hazel Blvd", region="north",
         created_at="2024-02-10T09:00:00Z",
         cancelled_at="2025-01-05T11:00:00Z"),
    dict(id="ORD-0108", subscriber_id="SUB-0007", plan_id=12, state="active",
         address="221B Maple St", region="west",
         created_at="2025-09-15T12:30:00Z", cancelled_at=None),
]

TECHNICIANS = [
    dict(id=200, full_name="Priya Raman", region="north"),
    dict(id=201, full_name="Marcus Webb", region="south"),
    dict(id=202, full_name="Sofia Reyes", region="west"),
]

# Future slot grid: for each technician, 3 working days x 3 windows.
# Ids embed 300 + index, assigned in (technician, day, window) order:
#   tech 200: Aug 3 -> SLT-0300..0302  Aug 4 -> SLT-0303..0305  Aug 5 -> SLT-0306..0308
#   tech 201: Aug 3 -> SLT-0309..0311  Aug 4 -> SLT-0312..0314  Aug 5 -> SLT-0315..0317
#   tech 202: Aug 3 -> SLT-0318..0320  Aug 4 -> SLT-0321..0323  Aug 5 -> SLT-0324..0326
_SLOT_DAYS = ["2026-08-03", "2026-08-04", "2026-08-05"]
_SLOT_WINDOWS = [("09:00", "11:00"), ("11:00", "13:00"), ("14:00", "16:00")]

# Extended grid: every remaining August weekday, appended AFTER the original
# grid in a separate id range (SLT-0400+) so the original ids, which task
# goals pin (SLT-0322, SLT-0326, ...), never shift. Deliberately no extra
# Aug 3-5 slots: "afternoon of August 5" must remain exactly one slot per
# region, or the reschedule-vlad goal state stops being deterministic.
_EXTENDED_DAYS = [
    "2026-08-06", "2026-08-07",
    "2026-08-10", "2026-08-11", "2026-08-12", "2026-08-13", "2026-08-14",
    "2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21",
    "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
    "2026-08-31",
]

# Slots referenced by seeded pending/completed appointments. Everything else
# is free, so every reschedule has real same-region alternatives.
_BOOKED_SLOT_IDS = {
    "SLT-0303", "SLT-0305", "SLT-0313", "SLT-0316", "SLT-0317", "SLT-0322",
    "SLT-0340",
}


def _slot_grid() -> list[dict]:
    slots = []
    n = 300
    for tech in TECHNICIANS:
        for day in _SLOT_DAYS:
            for start, end in _SLOT_WINDOWS:
                slot_id = f"SLT-{n:04d}"
                slots.append(dict(
                    id=slot_id,
                    technician_id=tech["id"],
                    starts_at=f"{day}T{start}:00Z",
                    ends_at=f"{day}T{end}:00Z",
                    status="booked" if slot_id in _BOOKED_SLOT_IDS
                    else "available",
                ))
                n += 1
    # One past slot backing Erin's completed installation.
    slots.append(dict(id="SLT-0340", technician_id=202,
                      starts_at="2026-07-10T09:00:00Z",
                      ends_at="2026-07-10T11:00:00Z",
                      status="booked"))
    # Extended August coverage, all free (demo: any weekday works).
    n = 400
    for tech in TECHNICIANS:
        for day in _EXTENDED_DAYS:
            for start, end in _SLOT_WINDOWS:
                slots.append(dict(
                    id=f"SLT-{n:04d}",
                    technician_id=tech["id"],
                    starts_at=f"{day}T{start}:00Z",
                    ends_at=f"{day}T{end}:00Z",
                    status="available",
                ))
                n += 1
    return slots


AVAILABILITY_SLOTS = _slot_grid()

APPOINTMENTS = [
    dict(id="APT-0400", order_id="ORD-0100", slot_id="SLT-0303",
         kind="maintenance", status="pending",
         notes="Intermittent fiber signal drops.",
         created_at="2026-07-20T13:05:00Z"),
    dict(id="APT-0401", order_id="ORD-0102", slot_id="SLT-0313",
         kind="maintenance", status="pending",
         notes="Router firmware refresh.",
         created_at="2026-07-22T10:40:00Z"),
    dict(id="APT-0402", order_id="ORD-0104", slot_id="SLT-0340",
         kind="installation", status="completed",
         notes="Initial install, completed on time.",
         created_at="2026-07-01T09:30:00Z"),
    dict(id="APT-0403", order_id="ORD-0101", slot_id=None,
         kind="maintenance", status="cancelled",
         notes="Customer cancelled visit; slot released.",
         created_at="2026-07-15T16:20:00Z"),
    dict(id="APT-0404", order_id="ORD-0103", slot_id="SLT-0316",
         kind="maintenance", status="pending",
         notes="Speed drops during evening hours.",
         created_at="2026-07-24T09:10:00Z"),
    dict(id="APT-0405", order_id="ORD-0101", slot_id="SLT-0305",
         kind="maintenance", status="pending",
         notes="Annual line check.",
         created_at="2026-07-27T14:45:00Z"),
    dict(id="APT-0406", order_id="ORD-0106", slot_id="SLT-0317",
         kind="installation", status="pending",
         notes="New line install at 9 Garnet Way.",
         created_at="2026-07-26T11:20:00Z"),
    dict(id="APT-0407", order_id="ORD-0108", slot_id="SLT-0322",
         kind="maintenance", status="pending",
         notes="Wi-Fi coverage check.",
         created_at="2026-07-23T15:00:00Z"),
]

INVOICES = [
    dict(id="INV-0500", order_id="ORD-0100", amount_cents=6999,
         issued_on="2026-06-01", due_on="2026-06-15", status="paid"),
    dict(id="INV-0501", order_id="ORD-0100", amount_cents=6999,
         issued_on="2026-07-01", due_on="2026-07-15", status="paid"),
    dict(id="INV-0502", order_id="ORD-0101", amount_cents=4999,
         issued_on="2026-07-01", due_on="2026-07-15", status="unpaid"),
    dict(id="INV-0503", order_id="ORD-0102", amount_cents=2999,
         issued_on="2026-06-05", due_on="2026-06-19", status="paid"),
    dict(id="INV-0504", order_id="ORD-0102", amount_cents=2999,
         issued_on="2026-07-05", due_on="2026-07-19", status="unpaid"),
    dict(id="INV-0505", order_id="ORD-0103", amount_cents=8999,
         issued_on="2026-07-10", due_on="2026-07-24", status="paid"),
    dict(id="INV-0506", order_id="ORD-0104", amount_cents=2999,
         issued_on="2026-07-12", due_on="2026-07-26", status="paid"),
    dict(id="INV-0507", order_id="ORD-0108", amount_cents=6999,
         issued_on="2026-07-03", due_on="2026-07-17", status="paid"),
]

# Retrieval corpus. Docs 600-607 are load-bearing rules, and 608-611 are
# distractors that make Recall@k meaningful. Eligibility rules live HERE,
# not in code and not in the system prompt.
POLICY_DOCUMENTS = [
    dict(id=600, slug="cancellation-confirmation", category="cancellation",
         title="Order cancellation requires explicit customer confirmation",
         body="Cancelling an order is a destructive action. Before an order "
              "is cancelled, the agent must ask the customer to confirm the "
              "cancellation and receive an affirmative response. A "
              "cancellation performed without a recorded confirmation "
              "request violates policy."),
    dict(id=601, slug="cancellation-unpaid-balance", category="cancellation",
         title="No cancellation with an unpaid balance",
         body="An order with unpaid invoices must not be cancelled. The "
              "outstanding balance must be settled first. If the order has "
              "any unpaid invoice, decline the cancellation and explain that "
              "the remaining balance has to be paid before the order can be "
              "closed."),
    dict(id=602, slug="plan-allowed-list", category="plan_change",
         title="Allowed target plans for service updates",
         body="A service update may only move an order to Fiber 500 Mbps, "
              "Fiber 1 Gig, or Fiber 2 Gig. Fiber 100 Mbps is a legacy plan "
              "closed to new activations; do not switch any order to it."),
    dict(id=603, slug="cancellation-appointment-cleanup", category="cancellation",
         title="Cancel open appointments before cancelling an order",
         body="Before an order is cancelled, every pending technician "
              "appointment on that order must be cancelled: mark the "
              "appointment cancelled, detach it from its technician slot, "
              "and release the slot so it becomes available to other "
              "customers."),
    dict(id=604, slug="plan-change-confirmation", category="plan_change",
         title="Service updates require customer confirmation",
         body="A service update (changing an order's plan) modifies the "
              "customer's bill and requires explicit customer confirmation "
              "before the new plan is applied."),
    dict(id=605, slug="appointment-order-state", category="scheduling",
         title="Appointment type must match the order state",
         body="Maintenance visits are only for active orders. Installation "
              "visits are only for orders pending installation. Do not book "
              "or keep visits on cancelled orders."),
    dict(id=606, slug="scheduling-region-match", category="scheduling",
         title="Technician region must match the order region",
         body="A technician appointment must be booked with a technician who "
              "serves the region of the order's address. Do not book slots "
              "belonging to technicians assigned to a different region."),
    dict(id=607, slug="scheduling-reschedule-slots", category="scheduling",
         title="Keep slot bookings consistent when rescheduling",
         body="When rescheduling an appointment, book the new technician "
              "slot, release the previously held slot so other customers can "
              "use it, and update the appointment to reference the new "
              "slot."),
    dict(id=608, slug="plan-change-billing-cycle", category="plan_change",
         title="Plan price takes effect next billing cycle",
         body="When a service update is applied, the new monthly price takes "
              "effect at the start of the next billing cycle. No proration "
              "is applied mid-cycle."),
    dict(id=609, slug="billing-equipment-return", category="billing",
         title="Equipment return after cancellation",
         body="After an order is cancelled, rental equipment such as routers "
              "and set-top boxes must be returned within 30 days to avoid a "
              "non-return fee."),
    dict(id=610, slug="billing-refunds", category="billing",
         title="Refund processing time",
         body="Refunds for closed orders are issued to the original payment "
              "method within 10 business days of the final invoice."),
    dict(id=611, slug="support-hours", category="general",
         title="Technician visit and support hours",
         body="Technician visits take place on weekdays between 09:00 and "
              "16:00 local time. Phone and chat support are available around "
              "the clock."),
]

# events starts empty: every event in a final state is agent-caused.
EVENTS: list[dict] = []


def seed_rows() -> dict[str, list[dict]]:
    """Rows per table, in FK-safe insertion order."""
    return {
        "subscribers": SUBSCRIBERS,
        "plans": PLANS,
        "technicians": TECHNICIANS,
        "orders": ORDERS,
        "availability_slots": AVAILABILITY_SLOTS,
        "appointments": APPOINTMENTS,
        "invoices": INVOICES,
        "policy_documents": POLICY_DOCUMENTS,
        "events": EVENTS,
    }
