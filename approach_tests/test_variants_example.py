"""LLM prompt-variation tests: generation guard, review gate, runner rotation.

As with the judge, the *wiring* is under test. The paraphrasing model is
scripted. The invariants: no variant that loses an entity survives the
guard, nothing unreviewed reaches an episode, and (G, D, P) is shared by
every phrasing.
"""

import json
from pathlib import Path

from telecom_aut.config import AgentConfig
from telecom_aut.testing import (
    append_variants_to_file,
    critical_literals,
    generate_variants,
    load_task,
    preserves_entities,
    run_task,
)
from telecom_aut.testing.fakes import ScriptedLLM

from .episodes import GOOD_CANCEL, run_scripted

TASKS = Path(__file__).parent / "tasks"


class TestEntityGuard:
    def test_critical_literals_are_derived_from_the_task(self):
        task = load_task(TASKS / "reschedule-vlad.json")
        required = critical_literals(task)
        assert required["names"] == ["Vlad Grybennikov"]
        assert required["numbers"] == ["5"]      # "August 5"
        assert required["ids"] == []             # instruction names no ids

    def test_variants_that_lose_entities_are_rejected(self):
        task = load_task(TASKS / "reschedule-vlad.json")
        required = critical_literals(task)
        assert preserves_entities(
            "Vlad Grybennikov needs his visit moved to Aug 5, afternoon.",
            required)
        assert preserves_entities(  # name survives case/separator changes
            "shift VLAD  GRYBENNIKOV's visit -> August 5 PM", required)
        assert not preserves_entities(
            "Move the subscriber's visit to the afternoon of August 5.",
            required)                             # lost the name
        assert not preserves_entities(
            "Move Vlad Grybennikov's visit to a later day.", required)  # lost the date

    def test_generation_filters_and_reports(self):
        task = load_task(TASKS / "reschedule-vlad.json")
        llm = ScriptedLLM([{"variants": [
            "Reschedule the maintenance visit for Vlad Grybennikov to the afternoon of August 5.",
            "Vlad Grybennikov: move visit -> Aug 5 afternoon window.",
            "Move the customer's visit to the afternoon.",          # no name/date
            task.instruction,                                       # not a variant
        ]}])
        accepted, rejected = generate_variants(task, llm, n=4)
        assert len(accepted) == 2
        assert len(rejected) == 2
        assert all("Vlad" in v for v in accepted)


class TestReviewGate:
    def test_written_variants_are_unreviewed_and_excluded(self, tmp_path):
        # The shipped file may already carry generated variants. The test
        # is about the DELTA an append produces, not a pristine file.
        src = TASKS / "cancel-vlad.json"
        copy = tmp_path / "cancel-vlad.json"
        copy.write_text(src.read_text())
        before = load_task(copy)
        text = "Please cancel the internet order for Vlad Grybennikov."

        append_variants_to_file(copy, [text])
        task = load_task(copy)
        assert len(task.variants) == len(before.variants) + 1
        added = task.variants[-1]
        assert added.text == text
        assert added.source == "llm"
        assert added.reviewed is False
        # Unreviewed -> not a phrasing the runner will use.
        assert text not in task.phrasings()
        assert task.phrasings() == before.phrasings()
        assert text in task.phrasings(include_unreviewed=True)

        # Idempotent: appending the same text twice stores it once.
        append_variants_to_file(copy, [text])
        assert len(load_task(copy).variants) == len(before.variants) + 1

    def test_reviewed_variants_join_the_phrasing_pool(self, tmp_path):
        copy = tmp_path / "cancel-vlad.json"
        copy.write_text((TASKS / "cancel-vlad.json").read_text())
        before = len(load_task(copy).phrasings())
        append_variants_to_file(copy, ["Cancel Vlad Grybennikov's order."])
        data = json.loads(copy.read_text())
        data["variants"][-1]["reviewed"] = True
        copy.write_text(json.dumps(data))
        assert len(load_task(copy).phrasings()) == before + 1


class TestRunnerRotation:
    def test_episodes_rotate_over_reviewed_phrasings(self):
        task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
        task.level3 = None
        from telecom_aut.testing.annotations import InstructionVariant
        task.variants = [
            InstructionVariant(text="Close Vlad Grybennikov's internet order.",
                               source="llm", reviewed=True),
            InstructionVariant(text="UNREVIEWED phrasing", reviewed=False),
        ]
        llm = ScriptedLLM(list(GOOD_CANCEL) * 3)
        outcome = run_task(task, AgentConfig(), k=3, llm=llm,
                           use_variants=True)
        used = [ep["instruction"] for ep in outcome["episodes"]]
        # canonical, reviewed variant, canonical again (never the unreviewed).
        assert used == [
            task.instruction,
            "Close Vlad Grybennikov's internet order.",
            task.instruction,
        ]
        # Same (G, D, P) for every phrasing: all three episodes pass.
        assert outcome["level1_successes"] == 3

    def test_variants_off_by_default(self):
        task = load_task(TASKS / "cancel-vlad.json").model_copy(deep=True)
        task.level3 = None
        outcome = run_task(task, AgentConfig(), k=1,
                           llm=ScriptedLLM(list(GOOD_CANCEL)))
        assert outcome["episodes"][0]["instruction"] == task.instruction
