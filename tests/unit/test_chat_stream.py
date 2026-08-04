"""Chat demo tool: the streaming emitter yields a sane event sequence.

Only run_streaming_episode is tested (with a scripted LLM). The HTTP layer is
demo plumbing, deliberately untested.
"""

from telecom_aut.agent import TelecomAgent
from telecom_aut.chat import compose_instruction, run_streaming_episode
from telecom_aut.config import AgentConfig
from telecom_aut.environment import TelecomEnv

from telecom_aut.testing.fakes import ScriptedLLM
from .test_agent_loop import RESCHEDULE_SCRIPT


def test_stream_emits_intent_slots_steps_final_and_returns_result():
    env = TelecomEnv.fresh()
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM(list(RESCHEDULE_SCRIPT)))
    events = []
    result = run_streaming_episode(
        agent, "Move Alice Nguyen's visit to August 5 afternoon.",
        events.append
    )
    env.close()

    types = [e["type"] for e in events]
    assert types[0] == "intent" and types[1] == "slots"
    assert types[-1] == "final"
    steps = [e for e in events if e["type"] == "step"]
    assert len(steps) == 7
    assert steps[0]["candidates"] == ["find_subscriber", "get_appointment"]
    assert events[-1]["status"] == "finished"
    assert result is not None and result.status == "finished"


def test_compose_instruction_passthrough_without_history():
    assert compose_instruction([], "what time is his appt?") == \
        "what time is his appt?"


def test_compose_instruction_folds_recent_exchanges():
    transcript = [("Find Vlad's orders", "Vlad has active order ORD-0108.")]
    composed = compose_instruction(transcript, "what time is his appt?")
    assert "Operator: Find Vlad's orders" in composed
    assert "Agent: Vlad has active order ORD-0108." in composed
    assert composed.rstrip().endswith("Current request: what time is his appt?")


def test_compose_instruction_caps_context_window():
    transcript = [(f"msg {i}", f"reply {i}") for i in range(10)]
    composed = compose_instruction(transcript, "latest")
    assert "msg 0" not in composed  # only the most recent 6 exchanges
    assert "msg 4" in composed and "msg 9" in composed


def test_stream_reports_llm_failure_as_error_event():
    env = TelecomEnv.fresh()
    agent = TelecomAgent(env, AgentConfig(), llm=ScriptedLLM([]))  # exhausts
    events = []
    result = run_streaming_episode(agent, "hello", events.append)
    env.close()
    assert events[-1]["type"] == "error"
    assert result is None
