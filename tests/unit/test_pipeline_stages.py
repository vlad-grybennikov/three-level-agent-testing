"""Pipeline-stage tests: each stage standalone, on frozen inputs.

No test here creates an environment, a database, or an agent loop.
Stage in, typed object out.
"""

import pytest

from telecom_aut.config import AgentConfig, SelectorConfig
from telecom_aut.pipeline import (
    FINISH_TOOL,
    BindingError,
    EntityExtractor,
    IntentClassifier,
    LLMArgumentBinder,
    SelectionInput,
    SlotSet,
    ToolSelector,
    load_binder_class,
)
from telecom_aut.tools import TOOL_NAMES

from telecom_aut.testing.fakes import ScriptedLLM

CFG = AgentConfig()

FROZEN_INPUT = SelectionInput(
    intent="reschedule_appointment",
    slots=SlotSet(subscriber_name="Alice Nguyen", target_date="2026-08-05"),
    history=['find_customer({"name_or_email": "Alice Nguyen"}) -> {"id": 1}'],
)


class TestStage1IntentClassifier:
    def test_classifies_on_frozen_utterance(self):
        llm = ScriptedLLM([{"intent": "cancel_order"}])
        result = IntentClassifier(llm, CFG).run("Cancel Bruno Silva's internet order.")
        assert result.intent == "cancel_order"
        assert len(llm.calls) == 1
        assert "Cancel Bruno Silva's internet order." in llm.calls[0][1]

    def test_classifies_view_appointments(self):
        llm = ScriptedLLM([{"intent": "view_appointments"}])
        result = IntentClassifier(llm, CFG).run("When is Carol Okafor's next visit?")
        assert result.intent == "view_appointments"

    @pytest.mark.parametrize("bad", [
        {"intent": "make_coffee"}, {"intent": 42}, {"wrong_key": "x"}, {},
    ])
    def test_unknown_output_maps_to_unsupported(self, bad):
        result = IntentClassifier(ScriptedLLM([bad]), CFG).run("do something")
        assert result.intent == "unsupported"

    def test_system_prompt_comes_from_config(self):
        custom = AgentConfig(system_prompt="CUSTOM PREAMBLE")
        llm = ScriptedLLM([{"intent": "service_update"}])
        IntentClassifier(llm, custom).run("switch Erin Walsh to fiber")
        assert llm.calls[0][0] == "CUSTOM PREAMBLE"  # surface #1 reaches stages


class TestStage2EntityExtractor:
    def test_extracts_typed_slots(self):
        llm = ScriptedLLM([{
            "subscriber_name": "Alice Nguyen", "appointment_id": "APT-0400",
            "target_date": "2026-08-05", "target_time_window": "afternoon",
            "subscriber_email": None, "order_id": None,
            "target_plan": None, "requested_field": None, "notes": None,
        }])
        slots = EntityExtractor(llm, CFG).run("move the subscriber's visit", "reschedule_appointment")
        assert slots.appointment_id == "APT-0400"
        assert slots.target_time_window == "afternoon"

    def test_malformed_slot_is_dropped_not_fatal(self):
        llm = ScriptedLLM([{
            "subscriber_name": "Alice Nguyen",
            "appointment_id": 400,  # wrong type (int, not APT-XXXX) -> dropped
        }])
        slots = EntityExtractor(llm, CFG).run("move the subscriber's visit", "reschedule_appointment")
        assert slots.subscriber_name == "Alice Nguyen"
        assert slots.appointment_id is None

    def test_invented_slot_names_are_ignored(self):
        llm = ScriptedLLM([{"subscriber_name": "Bob", "favourite_pizza": "hawaii"}])
        slots = EntityExtractor(llm, CFG).run("x", "cancel_order")
        assert slots.subscriber_name == "Bob"
        assert not hasattr(slots, "favourite_pizza")


class TestStage3ToolSelector:
    def test_ranks_topk_candidates_not_just_top1(self):
        llm = ScriptedLLM([{"ranking": [
            "get_appointment", "list_available_slots", "release_slot",
            "book_slot", "update_appointment", "list_plans",  # 6th: cut by k=5
        ]}])
        sel = ToolSelector(llm, CFG).run(FROZEN_INPUT)
        tools = [c.tool for c in sel.candidates]
        assert tools == ["get_appointment", "list_available_slots",
                         "release_slot", "book_slot", "update_appointment"]
        assert sel.top1 == "get_appointment"
        scores = [c.score for c in sel.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_hallucinated_and_duplicate_tools_are_filtered(self):
        llm = ScriptedLLM([{"ranking": [
            "warp_drive", "book_slot", "book_slot", "release_slot",
        ]}])
        sel = ToolSelector(llm, CFG).run(FROZEN_INPUT)
        assert [c.tool for c in sel.candidates] == ["book_slot", "release_slot"]

    def test_empty_ranking_falls_back_to_finish(self):
        sel = ToolSelector(ScriptedLLM([{"ranking": []}]), CFG).run(FROZEN_INPUT)
        assert sel.top1 == FINISH_TOOL

    def test_excluded_tools_leave_candidate_space(self):
        cfg = AgentConfig(selector=SelectorConfig(excluded_tools=["search_policy"]))
        llm = ScriptedLLM([{"ranking": ["search_policy", "list_invoices"]}])
        sel = ToolSelector(llm, cfg).run(FROZEN_INPUT)
        assert [c.tool for c in sel.candidates] == ["list_invoices"]
        assert "search_policy" not in llm.calls[0][1]  # not even offered

    def test_invert_ranking_flips_top1(self):
        cfg = AgentConfig(selector=SelectorConfig(invert_ranking=True))
        llm = ScriptedLLM([{"ranking": ["get_appointment", "book_slot"]}])
        sel = ToolSelector(llm, cfg).run(FROZEN_INPUT)
        assert sel.top1 == "book_slot"

    def test_repeated_history_lines_are_collapsed_in_prompt(self):
        from telecom_aut.pipeline.stages import _render_history

        line = 'list_appointments({"customer_id": 5}) -> [...]'
        rendered = _render_history(["a -> 1", line, line, line, "b -> 2"])
        assert rendered.count("list_appointments") == 1
        assert "[repeated 3x" in rendered
        assert rendered.splitlines()[0] == "a -> 1"  # singles untouched

    def test_selector_prompt_carries_anti_repeat_instruction(self):
        llm = ScriptedLLM([{"ranking": ["finish"]}])
        ToolSelector(llm, CFG).run(FROZEN_INPUT)
        assert "Never repeat a call" in llm.calls[0][1]
        assert "already FAILED" in llm.calls[0][1]  # bind-deadlock guidance

    def test_catalog_offers_all_tools_plus_finish(self):
        llm = ScriptedLLM([{"ranking": ["finish"]}])
        ToolSelector(llm, CFG).run(FROZEN_INPUT)
        catalog = llm.calls[0][1]
        for name in TOOL_NAMES + [FINISH_TOOL]:
            assert f"- {name}:" in catalog


class TestStage4ArgumentBinder:
    def test_binds_schema_valid_args(self):
        llm = ScriptedLLM([{"slot_id": "SLT-0304"}])
        call = LLMArgumentBinder(llm, CFG).run("book_slot", FROZEN_INPUT)
        assert call.tool == "book_slot" and call.args == {"slot_id": "SLT-0304"}

    def test_partial_update_survives_binding(self):
        llm = ScriptedLLM([{"appointment_id": "APT-0400", "slot_id": "SLT-0304"}])
        call = LLMArgumentBinder(llm, CFG).run("update_appointment", FROZEN_INPUT)
        assert call.args == {"appointment_id": "APT-0400", "slot_id": "SLT-0304"}
        assert "status" not in call.args  # unset fields are not forwarded

    def test_invalid_args_raise_binding_error(self):
        llm = ScriptedLLM([{"slot_id": "tomorrow"}])
        with pytest.raises(BindingError, match="slot_id"):
            LLMArgumentBinder(llm, CFG).run("book_slot", FROZEN_INPUT)

    def test_unknown_tool_raises_binding_error(self):
        with pytest.raises(BindingError, match="warp_drive"):
            LLMArgumentBinder(ScriptedLLM([]), CFG).run("warp_drive", FROZEN_INPUT)

    def test_binds_finish_pseudo_tool(self):
        llm = ScriptedLLM([{"summary": "All done."}])
        call = LLMArgumentBinder(llm, CFG).run(FINISH_TOOL, FROZEN_INPUT)
        assert call.args == {"summary": "All done."}

    def test_binder_class_is_loadable_from_dotted_path(self):
        cls = load_binder_class("telecom_aut.pipeline.stages:LLMArgumentBinder")
        assert cls is LLMArgumentBinder
        with pytest.raises(ImportError):
            load_binder_class("telecom_aut.pipeline.stages:NoSuchBinder")
