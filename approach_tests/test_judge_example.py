"""Classic LLM-as-judge tests: trajectory grading with its own pass^k.

The judge model is scripted. Under test is the wiring: the judge sees the
full trajectory evidence, verdicts aggregate into judge_pass_hat_k as a
column parallel to Level 1 (never replacing it), parsing fails closed, and
the judge is opt-in.
"""

from pathlib import Path

from telecom_aut.config import AgentConfig
from telecom_aut.testing import (
    build_judge_prompt,
    judge_episode_outcome,
    load_task,
    run_task,
    state_delta,
)
from telecom_aut.testing.fakes import ScriptedLLM

from .episodes import BAD_CANCEL, GOOD_CANCEL, run_scripted

TASKS = Path(__file__).parent / "tasks"


def test_state_delta_summarises_what_actually_changed():
    task = load_task(TASKS / "cancel-vlad.json")
    trace = run_scripted(BAD_CANCEL, task.instruction).trace
    # The lazy cancel changed exactly one projected cell.
    assert state_delta(trace) == [
        "orders/ORD-0108.state: 'active' -> 'cancelled'"
    ]


def test_judge_prompt_carries_the_full_trajectory_evidence():
    task = load_task(TASKS / "cancel-vlad.json")
    trace = run_scripted(BAD_CANCEL, task.instruction).trace
    system, user = build_judge_prompt(trace)
    assert "grading" in system or "evaluator" in system
    assert "refusal counts as SUCCESS" in system      # policy-aware rubric
    assert task.instruction in user
    assert 'cancel_order({"order_id": "ORD-0108"})' in user
    assert "orders/ORD-0108.state" in user            # what actually changed
    assert "Done, cancelled." in user                 # the reply under judgment
    # The skipped cleanup is visible to the judge by ABSENCE from the
    # actions/changes evidence: no update_appointment, no slot change.
    assert "update_appointment" not in user
    assert "SLT-0322" not in user


def test_judge_parses_verdicts_and_fails_closed():
    task = load_task(TASKS / "cancel-vlad.json")
    trace = run_scripted(GOOD_CANCEL, task.instruction).trace

    ok = judge_episode_outcome(trace, ScriptedLLM([
        {"passed": True, "reasoning": "Order cancelled with full cleanup."},
    ]))
    assert ok.passed and "cleanup" in ok.reasoning

    fail = judge_episode_outcome(trace, ScriptedLLM([
        {"passed": False, "reasoning": 42},           # sloppy types coerced
    ]))
    assert not fail.passed and fail.reasoning == "42"

    garbage = judge_episode_outcome(trace, ScriptedLLM([{"verdict": "?"}]))
    assert not garbage.passed                          # fail-closed


def test_judge_gets_its_own_pass_hat_k_column():
    """Classic pattern: judge verdicts aggregate independently of Level 1."""
    task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
    task.level3 = None
    # Three episodes: good, lazy, good. A scripted judge that catches the
    # lazy one (fails it) while passing the good ones.
    agent_llm = ScriptedLLM(
        list(GOOD_CANCEL) + list(BAD_CANCEL) + list(GOOD_CANCEL)
    )
    judge_llm = ScriptedLLM([
        {"passed": True, "reasoning": "done properly"},
        {"passed": False, "reasoning": "no policy check, visit stranded"},
        {"passed": True, "reasoning": "done properly"},
    ])
    outcome = run_task(task, AgentConfig(), k=3,
                       llm=agent_llm, judge_llm=judge_llm)
    assert outcome["judge_successes"] == 2
    assert outcome["judge_pass_hat_k"]["1"] == 0.6667
    assert outcome["judge_pass_hat_k"]["3"] == 0.0
    # Level 1 computed the same episodes independently (here they agree:
    # the agreement RATE between the columns is the reliability measurement).
    assert outcome["level1_successes"] == 2
    assert [ep["judge"]["passed"] for ep in outcome["episodes"]] == \
        [ep["level1_passed"] for ep in outcome["episodes"]]


def test_judge_and_level1_columns_are_independent():
    """A judge that grades everything wrong cannot move the oracle column.
    Disagreement is data, not contamination."""
    task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
    task.level3 = None
    hostile_judge = ScriptedLLM([{"passed": False, "reasoning": "no"}])
    outcome = run_task(task, AgentConfig(), k=1,
                       llm=ScriptedLLM(list(GOOD_CANCEL)),
                       judge_llm=hostile_judge)
    assert outcome["level1_successes"] == 1            # oracle: pass
    assert outcome["judge_successes"] == 0             # judge: fail
    assert outcome["pass_hat_k"]["1"] == 1.0
    assert outcome["judge_pass_hat_k"]["1"] == 0.0


def test_judge_errors_fail_closed_and_do_not_kill_the_run():
    """A truncated/unparseable judge reply is a failed verdict, not a crash
    (the regression that killed a live suite run on 2026-07-28)."""
    class ExplodingLLM:
        def complete_json(self, system, user):
            raise RuntimeError("truncated mid-JSON")

    task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
    task.level3 = None
    outcome = run_task(task, AgentConfig(), k=1,
                       llm=ScriptedLLM(list(GOOD_CANCEL)),
                       judge_llm=ExplodingLLM())
    ep = outcome["episodes"][0]
    assert ep["level1_passed"]                     # oracle column untouched
    assert not ep["judge"]["passed"]               # fail-closed
    assert "judge error" in ep["judge"]["reasoning"]
    assert outcome["judge_successes"] == 0


def test_judge_absent_by_default():
    task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
    task.level3 = None
    outcome = run_task(task, AgentConfig(), k=1,
                       llm=ScriptedLLM(list(GOOD_CANCEL)))
    assert outcome["episodes"][0]["judge"] is None
    assert outcome["judge_successes"] is None
    assert outcome["judge_pass_hat_k"] is None
