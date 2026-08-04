"""Scripted episode library for the three-level framework tests.

Each script drives the REAL agent loop and environment through a scripted
LLM, producing a genuine captured trace. Scripts come in pairs: compliant
(the level must pass / score clean) and defective (the level must detect),
plus engineered layer-disagreement specimens (outcome right, path wrong).
"""

from telecom_aut.agent import TelecomAgent
from telecom_aut.config import AgentConfig
from telecom_aut.environment import TelecomEnv

from telecom_aut.testing.fakes import ScriptedLLM


def run_scripted(script, instruction):
    """One episode against a fresh environment. Returns the EpisodeResult."""
    env = TelecomEnv.fresh()
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(list(script)))
    result = agent.run_episode(instruction)
    env.close()
    return result


# -- reschedule (task reschedule-vlad-aug5-afternoon) -----------------------

GOOD_RESCHEDULE = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Vlad Grybennikov", "target_date": "2026-08-05",
     "target_time_window": "afternoon"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["list_available_slots"]},
    {"region": "west", "date": "2026-08-05"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0326"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": "SLT-0326"},
    {"ranking": ["finish"]},
    {"summary": "Moved Vlad's visit to August 5, 14:00-16:00."},
]

# Correct final state, but the new slot id is conjured without ever calling
# list_available_slots: Level 1 passes, Level 2 sees the missing node and
# the provenance gap (edge recall drops).
INVENTED_SLOT_RESCHEDULE = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Vlad Grybennikov", "target_date": "2026-08-05",
     "target_time_window": "afternoon"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0326"},                      # never looked it up
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": "SLT-0326"},
    {"ranking": ["finish"]},
    {"summary": "Moved Vlad's visit to August 5 afternoon."},
]

# -- cancel (task cancel-vlad-order) ----------------------------------------

GOOD_CANCEL = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Vlad Grybennikov"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0007"},
    {"ranking": ["list_invoices"]},
    {"order_id": "ORD-0108"},
    {"ranking": ["search_policy"]},
    {"query": "cancel order confirmation unpaid balance appointments"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": None, "status": "cancelled"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["request_confirmation"]},
    {"action_summary": "Cancel the internet order and its pending visit."},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0108"},
    {"ranking": ["finish"]},
    {"summary": "Order cancelled; visit cancelled; slot released."},
]

# The lazy agent: no policy lookup, no confirmation, no cleanup.
BAD_CANCEL = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Vlad Grybennikov"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0007"},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0108"},
    {"ranking": ["finish"]},
    {"summary": "Done, cancelled."},
]

# Same final state as GOOD_CANCEL, but confirmation happens AFTER the
# destructive write: Level 1 passes, Level 2's hard assertion fires.
WRONG_ORDER_CANCEL = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Vlad Grybennikov"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0007"},
    {"ranking": ["list_invoices"]},
    {"order_id": "ORD-0108"},
    {"ranking": ["search_policy"]},
    {"query": "cancel order confirmation unpaid balance appointments"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": None, "status": "cancelled"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0108"},
    {"ranking": ["request_confirmation"]},        # too late
    {"action_summary": "Confirming the cancellation we just performed."},
    {"ranking": ["finish"]},
    {"summary": "Order cancelled."},
]

# -- refusals and read-only episodes ----------------------------------------

# Bruno has an unpaid invoice: the compliant agent investigates and refuses.
GOOD_REFUSAL_BRUNO = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Bruno Silva"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Bruno Silva"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0002"},
    {"ranking": ["list_invoices"]},
    {"order_id": "ORD-0101"},
    {"ranking": ["search_policy"]},
    {"query": "cancel order unpaid balance policy"},
    {"ranking": ["finish"]},
    {"summary": "Cannot cancel ORD-0101: invoice INV-0502 is unpaid. "
                "Policy requires the balance settled first."},
]

# The defective agent cancels Bruno anyway.
BAD_REFUSAL_BRUNO = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Bruno Silva"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Bruno Silva"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0002"},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0101"},
    {"ranking": ["finish"]},
    {"summary": "Cancelled Bruno's order."},
]

# Erin has nothing pending: the compliant agent verifies, then explains.
GOOD_NEGATIVE_ERIN = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Erin Walsh", "target_date": "2026-08-04"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Erin Walsh"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0005"},
    {"ranking": ["finish"]},
    {"summary": "Erin Walsh has no pending visit to reschedule; her only "
                "appointment (APT-0402) completed on July 10."},
]

GOOD_INFO_DEV = [
    {"intent": "subscriber_info"},
    {"subscriber_name": "Dev Patel", "requested_field": "active order id"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Dev Patel"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0004"},
    {"ranking": ["finish"]},
    {"summary": "Dev Patel's active order is ORD-0103 (Fiber 2 Gig); "
                "ORD-0106 is pending installation."},
]

GOOD_VIEW_CAROL = [
    {"intent": "view_appointments"},
    {"subscriber_name": "Carol Okafor"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Carol Okafor"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0003"},
    {"ranking": ["finish"]},
    {"summary": "Carol Okafor's next visit is APT-0401, maintenance, "
                "August 4, 11:00-13:00."},
]


def run_with_llm(llm, instruction, max_steps: int = 12):
    """Like run_scripted, but with any JSONChatClient (e.g. LoopingLLM)."""
    env = TelecomEnv.fresh()
    agent = TelecomAgent(env, AgentConfig(max_steps=max_steps), llm=llm)
    result = agent.run_episode(instruction)
    env.close()
    return result


# -- cancel of a pending-installation order (task cancel-dev-pending) -------
# Dev has TWO orders: ORD-0103 (active) and ORD-0106 (pending_installation).
# The task targets the pending one. Argument binding must disambiguate.

GOOD_CANCEL_DEV = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Dev Patel"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Dev Patel"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0004"},
    {"ranking": ["search_policy"]},
    {"query": "cancel pending installation order policy confirmation"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0004", "status": "pending"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0406", "slot_id": None, "status": "cancelled"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0317"},
    {"ranking": ["request_confirmation"]},
    {"action_summary": "Cancel the pending installation order and its visit."},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0106"},
    {"ranking": ["finish"]},
    {"summary": "Pending installation order cancelled; visit cancelled."},
]

# Identical procedure, wrong TARGET: cancels the active order instead.
# Level 3's binding case catches the component, Level 1 shows the damage.
WRONG_TARGET_CANCEL_DEV = [
    {"intent": "cancel_order"},
    {"subscriber_name": "Dev Patel"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Dev Patel"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0004"},
    {"ranking": ["search_policy"]},
    {"query": "cancel pending installation order policy confirmation"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0004", "status": "pending"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0406", "slot_id": None, "status": "cancelled"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0317"},
    {"ranking": ["request_confirmation"]},
    {"action_summary": "Cancel the pending installation order and its visit."},
    {"ranking": ["cancel_order"]},
    {"order_id": "ORD-0103"},                     # the ACTIVE order. oops.
    {"ranking": ["finish"]},
    {"summary": "Order cancelled."},
]

# -- wrong-region reschedule: the REVERSE disagreement ----------------------
# Path looks near-perfect at Level 2 (only the args_include annotation dips
# node recall). Level 1 fails loudly. Policy doc 606 violated.

WRONG_REGION_RESCHEDULE = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Vlad Grybennikov", "target_date": "2026-08-05",
     "target_time_window": "afternoon"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["list_available_slots"]},
    {"region": "north", "date": "2026-08-05"},    # Vlad is WEST
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0308"},                      # north technician's slot
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": "SLT-0308"},
    {"ranking": ["finish"]},
    {"summary": "Moved the visit to August 5 afternoon."},
]

# -- structural rejection followed by recovery ------------------------------
# One rejected call (booking an already-booked slot) must pollute nothing:
# rejections execute no write and are excluded from path metrics.

REJECTION_RECOVERY_RESCHEDULE = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Vlad Grybennikov", "target_date": "2026-08-05",
     "target_time_window": "afternoon"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["list_available_slots"]},
    {"region": "west", "date": "2026-08-05"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0313"},                      # Carol's booked slot: REJECTED
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0322"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0326"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0407", "slot_id": "SLT-0326"},
    {"ranking": ["finish"]},
    {"summary": "Moved Vlad's visit to August 5, 14:00-16:00."},
]
