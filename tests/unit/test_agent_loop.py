"""Agent-loop tests: LangGraph wiring driven by scripted fake LLMs.

These prove the loop executes stage outputs against the real environment,
enforces the step budget, and terminates, all offline and deterministic.
"""

import pytest

from telecom_aut.agent import TelecomAgent
from telecom_aut.config import AgentConfig
from telecom_aut.environment import TelecomEnv

from telecom_aut.testing.fakes import LoopingLLM, ScriptedLLM


@pytest.fixture()
def env():
    e = TelecomEnv.fresh()
    yield e
    e.close()


# One scripted reschedule episode: intent, slots, then (select, bind) pairs.
RESCHEDULE_SCRIPT = [
    {"intent": "reschedule_appointment"},
    {"subscriber_name": "Alice Nguyen", "target_date": "2026-08-05",
     "target_time_window": "afternoon"},
    {"ranking": ["find_subscriber", "get_appointment"]},
    {"name_or_email": "Alice Nguyen"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0001", "status": "pending"},
    {"ranking": ["list_available_slots"]},
    {"region": "north", "date": "2026-08-05"},
    {"ranking": ["release_slot"]},
    {"slot_id": "SLT-0303"},
    {"ranking": ["book_slot"]},
    {"slot_id": "SLT-0308"},
    {"ranking": ["update_appointment"]},
    {"appointment_id": "APT-0400", "slot_id": "SLT-0308"},
    {"ranking": ["finish"]},
    {"summary": "Moved appointment APT-0400 to the afternoon slot on August 5."},
]


def test_scripted_reschedule_episode_end_to_end(env):
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(RESCHEDULE_SCRIPT))
    result = agent.run_episode(
        "Reschedule Alice Nguyen's visit to the afternoon of August 5."
    )

    assert result.status == "finished"
    assert result.intent == "reschedule_appointment"
    assert len(result.steps) == 7

    # Final state is the goal state of a correct reschedule.
    snap = env.snapshot()["tables"]
    slots = {s["id"]: s["status"] for s in snap["availability_slots"]}
    appt = next(a for a in snap["appointments"] if a["id"] == "APT-0400")
    assert slots["SLT-0303"] == "available" and slots["SLT-0308"] == "booked"
    assert appt["slot_id"] == "SLT-0308" and appt["status"] == "pending"
    assert [e["operation"] for e in snap["events"]] == [
        "release_slot", "book_slot", "update_appointment",
    ]

    # Steps carry ranked candidates (top-k), not just the executed tool.
    first = result.steps[0]
    assert [c["tool"] for c in first["selection"]["candidates"]] == [
        "find_subscriber", "get_appointment",
    ]
    assert result.config_hash == AgentConfig().config_hash()


def test_view_appointments_episode_changes_nothing(env):
    """Read-only intent: the correct goal state is the initial state."""
    script = [
        {"intent": "view_appointments"},
        {"subscriber_name": "Carol Okafor"},
        {"ranking": ["find_subscriber"]},
        {"name_or_email": "Carol Okafor"},
        {"ranking": ["list_appointments"]},
        {"subscriber_id": "SUB-0003"},
        {"ranking": ["finish"]},
        {"summary": "Carol has one pending maintenance visit (APT-0401), "
                    "August 4, 11:00-13:00."},
    ]
    before = env.snapshot()
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(script))
    result = agent.run_episode("When is Carol Okafor's visit?")
    assert result.status == "finished"
    assert result.intent == "view_appointments"
    assert "APT-0401" in result.summary
    assert env.snapshot() == before  # zero writes, zero events, zero clock


def test_subscriber_info_field_level_question(env):
    """Op 1: 'What is the active order id for Erin?', answered in the
    finish summary, with no state change."""
    script = [
        {"intent": "subscriber_info"},
        {"subscriber_name": "Erin Walsh", "requested_field": "active order id"},
        {"ranking": ["find_subscriber"]},
        {"name_or_email": "Erin Walsh"},
        {"ranking": ["list_orders"]},
        {"subscriber_id": "SUB-0005"},
        {"ranking": ["finish"]},
        {"summary": "Erin Walsh's active order is ORD-0104 (Fiber 100 Mbps)."},
    ]
    before = env.snapshot()
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(script))
    result = agent.run_episode("What is the active order id for Erin Walsh?")
    assert result.status == "finished"
    assert result.intent == "subscriber_info"
    assert "ORD-0104" in result.summary
    assert env.snapshot() == before


def test_unconfirmed_cancellation_passes_through_the_loop(env):
    """The loop must not add policy safety the API doesn't have: a scripted
    agent that skips policy and confirmation cancels Bruno successfully."""
    script = [
        {"intent": "cancel_order"},
        {"subscriber_name": "Bruno Silva"},
        {"ranking": ["find_subscriber"]},
        {"name_or_email": "Bruno Silva"},
        {"ranking": ["list_orders"]},
        {"subscriber_id": "SUB-0002"},
        {"ranking": ["cancel_order"]},
        {"order_id": "ORD-0101"},
        {"ranking": ["finish"]},
        {"summary": "Your order ORD-0101 has been cancelled."},
    ]
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(script))
    result = agent.run_episode("Cancel Bruno Silva's internet order.")
    assert result.status == "finished"
    snap = env.snapshot()["tables"]
    order = next(o for o in snap["orders"] if o["id"] == "ORD-0101")
    assert order["state"] == "cancelled"
    # No request_confirmation event before the cancellation: the violation
    # is visible in final state, exactly as the study needs.
    assert [e["operation"] for e in snap["events"]] == ["cancel_order"]


def test_unsupported_intent_short_circuits(env):
    agent = TelecomAgent(
        env, AgentConfig(), llm=ScriptedLLM([{"intent": "unsupported"}])
    )
    result = agent.run_episode("What's the meaning of life?")
    assert result.status == "unsupported"
    assert result.steps == []
    assert env.snapshot()["tables"]["events"] == []


def test_step_budget_is_enforced(env):
    # Selector forever picks list_plans, binder always returns {}.
    llm = LoopingLLM([{"intent": "service_update"}, {},  # intent, slots
                      {"ranking": ["list_plans"]}, {}])
    agent = TelecomAgent(env, AgentConfig(max_steps=3), llm=llm)
    result = agent.run_episode("Switch Dev Patel to a bigger plan.")
    assert result.status == "budget_exhausted"
    assert len(result.steps) == 3
    assert env.snapshot()["tables"]["events"] == []  # reads only


def test_binding_failure_is_recorded_and_loop_continues(env):
    script = [
        {"intent": "reschedule_appointment"},
        {"subscriber_name": "Alice Nguyen"},
        {"ranking": ["get_appointment"]},
        {"appointment_id": "not-an-id"},          # pattern fails -> recorded
        {"ranking": ["get_appointment"]},
        {"appointment_id": "APT-0400"},           # retry succeeds
        {"ranking": ["finish"]},
        {"summary": "Checked the appointment."},
    ]
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(script))
    result = agent.run_episode("When is Alice Nguyen's visit? Move it"
                                " if needed.")
    assert result.status == "finished"
    assert len(result.steps) == 3
    assert result.steps[0]["error"].startswith("binding failed")
    assert result.steps[0]["call"] is None
    assert result.steps[1]["result"]["id"] == "APT-0400"


def test_tool_rejection_feeds_back_into_history(env):
    script = [
        {"intent": "reschedule_appointment"},
        {"subscriber_name": "Alice Nguyen"},
        {"ranking": ["book_slot"]},
        {"slot_id": "SLT-0303"},              # already booked -> REJECTED
        {"ranking": ["finish"]},
        {"summary": "Could not book that slot."},
    ]
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(script))
    result = agent.run_episode("Rebook the same slot for Alice Nguyen.")
    assert result.status == "finished"
    rejected = result.steps[0]
    assert rejected["error"] is not None and "invalid_state" in rejected["error"]
    # The rejection left no state behind (structural guarantee).
    assert env.snapshot()["tables"]["events"] == []


def test_identical_scripts_produce_identical_state_and_steps():
    def run():
        env = TelecomEnv.fresh()
        agent = TelecomAgent(
            env, AgentConfig(), llm=ScriptedLLM(list(RESCHEDULE_SCRIPT))
        )
        result = agent.run_episode("Move Alice's visit to August 5 afternoon.")
        snap = env.snapshot()
        env.close()
        dump = result.model_dump()
        dump["trace"].pop("timing")  # wall-clock, everything else is frozen
        return dump, snap

    r1, s1 = run()
    r2, s2 = run()
    assert r1 == r2 and s1 == s2


def test_trace_interleaves_snapshots_with_calls(env):
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(RESCHEDULE_SCRIPT))
    result = agent.run_episode("Move Vlad Grybennikov's visit to August 5"
                                " afternoon.")
    trace = result.trace
    assert trace["trace_schema_version"] == 1
    assert trace["initial_snapshot"]["tables"]["events"] == []
    # Every executed tool call carries a post-call snapshot, finish doesn't.
    tool_steps = [s for s in trace["steps"] if s["call"] and
                  s["call"]["tool"] != "finish"]
    assert all(s["post_call_snapshot"] is not None for s in tool_steps)
    assert trace["steps"][-1]["post_call_snapshot"] is None  # finish
    # Snapshots show state evolving: release_slot's snapshot has 1 event.
    first_write = next(s for s in trace["steps"]
                       if s["call"]["tool"] == "release_slot")
    assert len(first_write["post_call_snapshot"]["tables"]["events"]) == 1
    assert trace["final_snapshot"] == agent.env.snapshot()
    assert trace["config_hash"] == result.config_hash
