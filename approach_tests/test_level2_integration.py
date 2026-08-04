"""Level 2, integration (§3.3): the run's path against D, plus P.

Covers: clean scores on compliant paths, fault localisation on defective
ones, and the two engineered layer-disagreement specimens (outcome right /
path wrong) that motivate the level's existence.
"""

from pathlib import Path

import pytest

from telecom_aut.testing import evaluate_trace, load_task

from .episodes import (
    BAD_CANCEL,
    BAD_REFUSAL_BRUNO,
    GOOD_CANCEL,
    GOOD_REFUSAL_BRUNO,
    GOOD_RESCHEDULE,
    INVENTED_SLOT_RESCHEDULE,
    WRONG_ORDER_CANCEL,
    run_scripted,
)

TASKS = Path(__file__).parent / "tasks"


def level2_for(script, task_file):
    task = load_task(TASKS / task_file)
    result = run_scripted(script, task.instruction)
    verdict = evaluate_trace(result.trace, task)
    return verdict, task


class TestCompliantPaths:
    def test_cancel_chain_scores_clean(self):
        verdict, _ = level2_for(GOOD_CANCEL, "cancel-vlad.json")
        l2 = verdict["level2"]
        assert l2["node_recall"] == 1.0 and l2["node_precision"] == 1.0
        assert l2["edge_recall"] == 1.0 and l2["edge_precision"] == 1.0
        assert l2["order_conformance"] == 1.0
        assert l2["redundancy_ratio"] == 0.0
        assert all(p["passed"] for p in l2["policy_results"])

    def test_reschedule_chain_scores_clean(self):
        verdict, _ = level2_for(GOOD_RESCHEDULE, "reschedule-vlad.json")
        l2 = verdict["level2"]
        assert l2["node_f1"] == 1.0 and l2["edge_f1"] == 1.0
        assert l2["order_conformance"] == 1.0

    def test_compliant_refusal_scores_clean(self):
        verdict, _ = level2_for(GOOD_REFUSAL_BRUNO, "cancel-bruno-refusal.json")
        l2 = verdict["level2"]
        assert l2["node_recall"] == 1.0
        assert all(p["passed"] for p in l2["policy_results"])  # forbid_call


class TestFaultLocalisation:
    def test_lazy_cancel_names_the_missing_operations(self):
        verdict, _ = level2_for(BAD_CANCEL, "cancel-vlad.json")
        l2 = verdict["level2"]
        assert l2["node_recall"] < 0.5
        assert set(l2["missing_ops"]) >= {
            "search_policy", "request_confirmation", "release_slot",
            "update_appointment", "list_invoices",
        }
        assert [p["passed"] for p in l2["policy_results"]] == [False, False]
        assert l2["order_conformance"] == 0.0

    def test_forbidden_write_is_a_hard_failure(self):
        verdict, _ = level2_for(BAD_REFUSAL_BRUNO, "cancel-bruno-refusal.json")
        l2 = verdict["level2"]
        forbid = [p for p in l2["policy_results"]
                  if p["assertion"]["kind"] == "forbid_call"]
        assert forbid and not forbid[0]["passed"]
        assert "cancel_order" in forbid[0]["detail"]


class TestLayerDisagreementSpecimens:
    """Outcome identical to the compliant run. Only the path differs.
    These episodes are the paper's core exhibit: Level 1 passes, Level 2
    detects."""

    def test_confirmation_after_the_fact(self):
        verdict, _ = level2_for(WRONG_ORDER_CANCEL, "cancel-vlad.json")
        assert verdict["level1_passed"]            # state is exactly G
        l2 = verdict["level2"]
        assert l2["node_recall"] == 1.0            # every op present...
        confirm = [p for p in l2["policy_results"]
                   if p["assertion"]["first"] == "request_confirmation"]
        assert not confirm[0]["passed"]            # ...but too late
        assert l2["order_conformance"] == 0.75     # 3 of 4 pairs hold

    def test_invented_slot_id_shows_as_provenance_gap(self):
        verdict, _ = level2_for(INVENTED_SLOT_RESCHEDULE,
                                "reschedule-vlad.json")
        assert verdict["level1_passed"]            # final state is correct
        l2 = verdict["level2"]
        assert "list_available_slots" in l2["missing_ops"]
        # Declared edges through the missing lookup cannot be recovered:
        assert l2["edge_recall"] == 0.6            # 3 of 5 declared edges
        # And the conjured value's only provenance is book_slot's own echo,
        # an edge the annotation never declared:
        assert l2["edge_precision"] == 0.75


class TestRedundancy:
    def test_verbatim_repeat_is_flagged_but_outcome_passes(self):
        script = list(GOOD_RESCHEDULE)
        script[4:4] = [{"ranking": ["list_appointments"]},
                       {"subscriber_id": "SUB-0007", "status": "pending"}]
        verdict, _ = level2_for(script, "reschedule-vlad.json")
        assert verdict["level1_passed"]            # Level 1 is blind to waste
        assert verdict["level2"]["redundancy_ratio"] > 0.0


class TestExtendedPathCoverage:
    def test_wrong_region_is_the_reverse_disagreement(self):
        """Level 1 fails loudly, Level 2 stays structurally clean: the
        tool sequence, edges, and ordering are all perfect. Only the two
        args_include annotations (west lookup, SLT-0326 booking) dip node
        recall. Tool-name edges cannot see the wrong region at all."""
        from .episodes import WRONG_REGION_RESCHEDULE
        verdict, _ = level2_for(WRONG_REGION_RESCHEDULE,
                                "reschedule-vlad.json")
        assert not verdict["level1_passed"]
        l2 = verdict["level2"]
        assert l2["node_recall"] == pytest.approx(4 / 6, abs=1e-3)
        assert {"list_available_slots", "book_slot"} <= set(l2["missing_ops"])
        assert l2["edge_recall"] == 1.0 and l2["edge_precision"] == 1.0
        assert l2["order_conformance"] == 1.0

    def test_wrong_target_cancel_is_localised_to_the_argument(self):
        from .episodes import WRONG_TARGET_CANCEL_DEV
        verdict, _ = level2_for(WRONG_TARGET_CANCEL_DEV,
                                "cancel-dev-pending.json")
        l2 = verdict["level2"]
        # Only the args_include on cancel_order misses. Procedure is intact.
        assert l2["node_recall"] == pytest.approx(7 / 8, abs=1e-3)
        assert l2["order_conformance"] == 1.0
        assert all(p["passed"] for p in l2["policy_results"])

    def test_rejected_calls_do_not_pollute_path_metrics(self):
        from .episodes import REJECTION_RECOVERY_RESCHEDULE
        verdict, task = level2_for(REJECTION_RECOVERY_RESCHEDULE,
                                   "reschedule-vlad.json")
        l2 = verdict["level2"]
        assert l2["node_f1"] == 1.0 and l2["edge_f1"] == 1.0
        assert l2["redundancy_ratio"] == 0.0
        # The rejection is still visible in the trace itself.
        assert "book_slot" in l2["observed_ops"]

    def test_stalled_episode_floods_redundancy(self):
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
        l2 = evaluate_trace(result.trace, task)["level2"]
        assert l2["redundancy_ratio"] == pytest.approx(5 / 6, abs=1e-3)
        # 6 successful identical list_orders calls (interleaved bind
        # failures execute nothing and are excluded from path metrics)
        assert l2["node_recall"] < 0.2
