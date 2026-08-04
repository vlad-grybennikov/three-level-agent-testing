"""Level 3, components in isolation (§3.4): frozen inputs, typed outputs.

Four decision components, each judged separately: intent classification
(per-class F1 over a labeled set), entity extraction (per-slot match), tool
selection (Recall@k over the exposed ranking), argument binding (schema
validity + exact match). No loop, no database mutation.
"""

from pathlib import Path

import pytest

from telecom_aut.config import AgentConfig
from telecom_aut.testing import (
    evaluate_intent_classification,
    evaluate_level3,
    load_intent_dataset,
    load_task,
)

from telecom_aut.testing.fakes import ScriptedLLM

ROOT = Path(__file__).parent
TASKS = ROOT / "tasks"
CFG = AgentConfig()


class TestIntentClassificationF1:
    CASES_JSON = [
        {"utterance": "What plan is Dev Patel on?", "expected": "subscriber_info"},
        {"utterance": "Show Frank Osei's account state.", "expected": "subscriber_info"},
        {"utterance": "When is Carol Okafor's next visit?", "expected": "view_appointments"},
        {"utterance": "List Bruno Silva's appointments.", "expected": "view_appointments"},
        {"utterance": "Cancel Bruno Silva's internet order.", "expected": "cancel_order"},
        {"utterance": "Carol Okafor wants to close her account.", "expected": "cancel_order"},
        {"utterance": "What's the weather in the north region?", "expected": "unsupported"},
        {"utterance": "Give Bruno Silva a discount.", "expected": "unsupported"},
    ]

    def _report(self, predictions):
        from telecom_aut.testing import IntentCase
        cases = [IntentCase.model_validate(c) for c in self.CASES_JSON]
        llm = ScriptedLLM([{"intent": p} for p in predictions])
        return evaluate_intent_classification(cases, CFG, llm)

    def test_perfect_classifier_scores_one(self):
        report = self._report([c["expected"] for c in self.CASES_JSON])
        assert report.accuracy == 1.0 and report.macro_f1 == 1.0
        assert report.errors == []
        assert all(m.f1 == 1.0 for m in report.per_class.values())

    def test_confusions_produce_per_class_f1(self):
        # Two engineered errors: a view->info confusion and an unsupported
        # request accepted as cancel_order.
        report = self._report([
            "subscriber_info", "subscriber_info",
            "view_appointments", "subscriber_info",   # error 1
            "cancel_order", "cancel_order",
            "unsupported", "cancel_order",            # error 2
        ])
        assert report.accuracy == 0.75
        assert report.macro_f1 == pytest.approx(0.7333, abs=1e-3)
        info = report.per_class["subscriber_info"]
        assert info.precision == pytest.approx(2 / 3, abs=1e-3)
        assert info.recall == 1.0
        view = report.per_class["view_appointments"]
        assert view.recall == 0.5 and view.precision == 1.0
        assert len(report.errors) == 2
        assert report.errors[0]["expected"] == "view_appointments"

    def test_shipped_dataset_loads_and_covers_all_intents(self):
        cases = load_intent_dataset(ROOT / "fixtures" / "level3_intents.json")
        labels = {c.expected for c in cases}
        assert labels == {
            "subscriber_info", "view_appointments", "reschedule_appointment",
            "service_update", "cancel_order", "unsupported",
        }
        assert len(cases) >= 18  # 3 per class


class TestPerTaskComponentFixtures:
    def test_all_components_clean_on_reschedule_task(self):
        task = load_task(TASKS / "reschedule-vlad.json")
        llm = ScriptedLLM([
            {"intent": "reschedule_appointment"},
            {"subscriber_name": "Vlad Grybennikov",
             "target_date": "2026-08-05", "target_time_window": "afternoon"},
            {"ranking": ["find_subscriber", "list_orders"]},      # case 1
            {"ranking": ["list_appointments", "get_appointment"]},  # case 2
            {"slot_id": "SLT-0322"},                              # bind 1
            {"slot_id": "SLT-0326"},                              # bind 2
        ])
        l3 = evaluate_level3(task, CFG, llm)
        assert l3.intent_correct
        assert l3.slot_accuracy == 1.0
        assert l3.selection_recall_at_k == 1.0
        assert l3.binding_valid == [True, True]
        assert l3.binding_exact == [True, True]
        assert l3.binding_accuracy == 1.0

    def test_each_component_fault_is_attributed_separately(self):
        task = load_task(TASKS / "reschedule-vlad.json")
        llm = ScriptedLLM([
            {"intent": "cancel_order"},                       # stage 1 wrong
            {"subscriber_name": "Vlad Grybennikov"},          # date missed
            {"ranking": ["find_subscriber"]},                 # case 1 ok
            {"ranking": ["list_plans", "search_policy"]},     # case 2 miss
            {"slot_id": "the Tuesday slot"},                  # bind 1 invalid
            {"slot_id": "SLT-0325"},                          # bind 2 wrong slot
        ])
        l3 = evaluate_level3(task, CFG, llm)
        assert not l3.intent_correct
        assert l3.predicted_intent == "cancel_order"
        assert l3.slot_results == {
            "subscriber_name": True, "target_date": False,
            "target_time_window": False,
        }
        assert l3.selection_hits == [True, False]
        assert l3.selection_recall_at_k == 0.5
        assert l3.binding_valid == [False, True]   # invalid vs valid-but-wrong
        assert l3.binding_exact == [False, False]
        assert l3.binding_accuracy == 0.0

    def test_slot_matching_is_format_tolerant(self):
        # "Fiber 500" (annotation) must accept "fiber-500" (model output).
        task = load_task(TASKS / "upgrade-erin.json")
        llm = ScriptedLLM([
            {"intent": "service_update"},
            {"subscriber_name": "erin walsh", "target_plan": "fiber-500"},
            {"ranking": ["find_subscriber"]},
            {"order_id": "ORD-0104", "new_plan_code": "fiber-500"},
        ])
        l3 = evaluate_level3(task, CFG, llm)
        assert l3.slot_results["subscriber_name"] is True   # case-insensitive
        assert l3.slot_results["target_plan"] is True       # containment
        assert l3.binding_exact == [True]

    def test_binder_respects_configured_binder_class(self):
        # Level 3 must evaluate the binder named in config (surface #5),
        # not a hardcoded one. A bogus path must fail loudly, not silently
        # fall back to the default binder.
        task = load_task(TASKS / "cancel-vlad.json")
        cfg = AgentConfig(
            binder_class="telecom_aut.pipeline.stages:NoSuchBinder"
        )
        llm = ScriptedLLM([
            {"intent": "cancel_order"},
            {"subscriber_name": "Vlad Grybennikov"},
            {"ranking": ["find_subscriber"]},
        ])
        with pytest.raises(ImportError):
            evaluate_level3(task, cfg, llm)

    def test_binder_disambiguates_multi_order_subscriber(self):
        # Dev has an active AND a pending order. The task targets the
        # pending one. Exact-match binding catches the classic wrong-id.
        task = load_task(TASKS / "cancel-dev-pending.json")
        good = ScriptedLLM([
            {"intent": "cancel_order"}, {"subscriber_name": "Dev Patel"},
            {"ranking": ["find_subscriber"]},
            {"order_id": "ORD-0106"},
        ])
        l3 = evaluate_level3(task, CFG, good)
        assert l3.binding_valid == [True] and l3.binding_exact == [True]

        bad = ScriptedLLM([
            {"intent": "cancel_order"}, {"subscriber_name": "Dev Patel"},
            {"ranking": ["find_subscriber"]},
            {"order_id": "ORD-0103"},          # valid schema, wrong order
        ])
        l3 = evaluate_level3(task, CFG, bad)
        assert l3.binding_valid == [True]      # schema cannot catch this
        assert l3.binding_exact == [False]     # exact-match does
        assert l3.binding_accuracy == 0.0

    def test_every_shipped_task_has_level3_fixtures(self):
        from telecom_aut.testing import load_tasks
        tasks = load_tasks(TASKS)
        assert len(tasks) == 9
        assert all(t.level3 is not None for t in tasks)
        capabilities = {t.capability for t in tasks}
        assert capabilities == {
            "subscriber_info", "view_appointments", "reschedule_appointment",
            "service_update", "cancel_order",
        }
