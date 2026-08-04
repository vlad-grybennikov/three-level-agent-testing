"""Level 1, end-to-end outcome (§3.2): projected final state vs G, pass^k.

Success and failure coverage per capability class: mutating happy paths,
policy refusals (correct outcome = state unchanged), read-only negative
controls, and the pass^k aggregation over stochastic-style mixed runs.
"""

from pathlib import Path

import pytest

from telecom_aut.config import AgentConfig
from telecom_aut.testing import evaluate_trace, load_task, pass_hat_k, run_task

from telecom_aut.testing.fakes import ScriptedLLM
from .episodes import (
    BAD_CANCEL,
    BAD_REFUSAL_BRUNO,
    GOOD_CANCEL,
    GOOD_INFO_DEV,
    GOOD_NEGATIVE_ERIN,
    GOOD_REFUSAL_BRUNO,
    GOOD_RESCHEDULE,
    GOOD_VIEW_CAROL,
    run_scripted,
)

TASKS = Path(__file__).parent / "tasks"


def verdict_for(script, task_file):
    task = load_task(TASKS / task_file)
    result = run_scripted(script, task.instruction)
    return evaluate_trace(result.trace, task), task


class TestMutatingHappyPaths:
    def test_reschedule_reaches_goal_state(self):
        verdict, _ = verdict_for(GOOD_RESCHEDULE, "reschedule-vlad.json")
        assert verdict["level1_passed"], verdict["level1"]["diffs"]

    def test_cancel_reaches_goal_state(self):
        verdict, _ = verdict_for(GOOD_CANCEL, "cancel-vlad.json")
        assert verdict["level1_passed"], verdict["level1"]["diffs"]

    def test_lazy_cancel_fails_with_localised_diffs(self):
        verdict, _ = verdict_for(BAD_CANCEL, "cancel-vlad.json")
        assert not verdict["level1_passed"]
        diffs = "\n".join(verdict["level1"]["diffs"])
        # Stranded appointment and slot are visible in state. The order
        # itself WAS cancelled, so it must not appear in the diffs.
        assert "APT-0407" in diffs and "SLT-0322" in diffs
        assert "ORD-0108" not in diffs


class TestRefusalsAndNegativeControls:
    """Correct behaviour here is NOT acting: G is empty."""

    def test_compliant_refusal_passes(self):
        verdict, _ = verdict_for(GOOD_REFUSAL_BRUNO, "cancel-bruno-refusal.json")
        assert verdict["level1_passed"]

    def test_violating_refusal_fails(self):
        verdict, _ = verdict_for(BAD_REFUSAL_BRUNO, "cancel-bruno-refusal.json")
        assert not verdict["level1_passed"]
        assert any("ORD-0101" in d for d in verdict["level1"]["diffs"])

    def test_nothing_to_reschedule_negative_control(self):
        verdict, _ = verdict_for(GOOD_NEGATIVE_ERIN,
                                 "reschedule-erin-negative.json")
        assert verdict["level1_passed"]

    def test_read_only_info_task_passes(self):
        verdict, _ = verdict_for(GOOD_INFO_DEV, "info-dev-active-order.json")
        assert verdict["level1_passed"]

    def test_read_only_view_task_passes(self):
        verdict, _ = verdict_for(GOOD_VIEW_CAROL, "view-carol-visit.json")
        assert verdict["level1_passed"]

    def test_any_mutation_fails_an_empty_goal(self):
        # An empty G means "change nothing": a full reschedule against it
        # must fail even though every individual write was well-formed.
        task = load_task(TASKS / "reschedule-vlad.json").model_copy(deep=True)
        task.goal.expected = {}
        result = run_scripted(GOOD_RESCHEDULE, task.instruction)
        assert not evaluate_trace(result.trace, task)["level1_passed"]


class TestPassHatK:
    def test_estimator_math(self):
        assert pass_hat_k(4, 3, 1) == 0.75          # plain pass rate
        assert pass_hat_k(4, 3, 2) == 0.5           # C(3,2)/C(4,2)
        assert pass_hat_k(4, 4, 4) == 1.0
        assert pass_hat_k(4, 2, 3) == 0.0
        with pytest.raises(ValueError):
            pass_hat_k(2, 2, 3)

    def test_run_task_aggregates_mixed_outcomes(self):
        """k=4 episodes through the real runner: 3 succeed, 1 is the lazy
        cancel. pass^k must degrade exactly as the estimator predicts."""
        task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
        task.level3 = None  # component pass not under test here
        llm = ScriptedLLM(
            list(GOOD_CANCEL) + list(GOOD_CANCEL)
            + list(BAD_CANCEL) + list(GOOD_CANCEL)
        )
        outcome = run_task(task, AgentConfig(), k=4, llm=llm)
        assert outcome["level1_successes"] == 3
        assert outcome["pass_hat_k"] == {
            "1": 0.75, "2": 0.5, "3": 0.25, "4": 0.0,
        }
        # Every episode carries its own full trace for the report.
        assert all("trace" in ep for ep in outcome["episodes"])
        statuses = [ep["level1_passed"] for ep in outcome["episodes"]]
        assert statuses == [True, True, False, True]


class TestExtendedOutcomeCoverage:
    def test_pending_installation_cancel_reaches_goal(self):
        from .episodes import GOOD_CANCEL_DEV
        verdict, _ = verdict_for(GOOD_CANCEL_DEV, "cancel-dev-pending.json")
        assert verdict["level1_passed"], verdict["level1"]["diffs"]

    def test_wrong_target_cancel_names_both_orders(self):
        """Binding fault on a multi-order subscriber: the diff names both
        the order that should have changed and the one that shouldn't."""
        from .episodes import WRONG_TARGET_CANCEL_DEV
        verdict, _ = verdict_for(WRONG_TARGET_CANCEL_DEV,
                                 "cancel-dev-pending.json")
        assert not verdict["level1_passed"]
        diffs = "\n".join(verdict["level1"]["diffs"])
        assert "ORD-0106" in diffs   # still pending_installation
        assert "ORD-0103" in diffs   # collateral cancellation

    def test_wrong_region_booking_fails_the_goal(self):
        from .episodes import WRONG_REGION_RESCHEDULE
        verdict, _ = verdict_for(WRONG_REGION_RESCHEDULE,
                                 "reschedule-vlad.json")
        assert not verdict["level1_passed"]
        diffs = "\n".join(verdict["level1"]["diffs"])
        assert "SLT-0326" in diffs and "SLT-0308" in diffs

    def test_rejected_call_then_recovery_still_passes(self):
        from .episodes import REJECTION_RECOVERY_RESCHEDULE
        verdict, _ = verdict_for(REJECTION_RECOVERY_RESCHEDULE,
                                 "reschedule-vlad.json")
        assert verdict["level1_passed"]

    def test_budget_exhaustion_fails_a_mutating_goal(self):
        from telecom_aut.testing.fakes import LoopingLLM
        from .episodes import run_with_llm
        task = load_task(TASKS / "cancel-vlad.json")
        llm = LoopingLLM([
            {"intent": "cancel_order"},
            {"subscriber_name": "Vlad Grybennikov"},
            {"ranking": ["list_orders"]},
            {"subscriber_id": "SUB-0007"},
        ])
        result = run_with_llm(llm, task.instruction)
        assert result.status == "budget_exhausted"
        assert not evaluate_trace(result.trace, task)["level1_passed"]
