"""Intent-dataset paraphrase enrichment: guard, review gate, idempotency.

As with the other LLM-assisted tooling, the paraphrasing model is scripted.
The invariants under test are the wiring: labels never change, entity-losing
paraphrases never survive, unreviewed entries never reach an evaluation,
and author cases are the only paraphrase sources.
"""

import json
from pathlib import Path

from telecom_aut.testing import (
    augment_dataset_file,
    critical_literals_for_utterance,
    generate_intent_paraphrases,
    load_intent_dataset,
)
from telecom_aut.testing.annotations import IntentCase
from telecom_aut.testing.fakes import ScriptedLLM
from telecom_aut.testing.intent_augment import PARAPHRASE_SYSTEM_PROMPT

FIXTURE = Path(__file__).parent / "fixtures" / "level3_intents.json"


class TestEntityGuard:
    def test_literals_derived_from_utterance_text(self):
        required = critical_literals_for_utterance(
            "Reschedule APT-0404 to August 3."
        )
        assert required["ids"] == ["APT-0404"]
        assert required["numbers"] == ["0404", "3"]
        required = critical_literals_for_utterance(
            "Cancel Bruno Silva's internet order."
        )
        assert required["names"] == ["Bruno Silva"]

    def test_paraphrases_that_lose_entities_are_rejected(self):
        case = IntentCase(utterance="Cancel Bruno Silva's internet order.",
                          expected="cancel_order")
        llm = ScriptedLLM([{"paraphrases": [
            "Please close the internet order for Bruno Silva.",
            "bruno silva wants his internet order terminated",  # case-tolerant
            "Close the customer's internet order.",             # lost the name
        ]}])
        accepted, rejected = generate_intent_paraphrases(case, llm, n=3)
        assert len(accepted) == 2
        assert rejected == ["Close the customer's internet order."]
        # The prompt never asks the model to label anything.
        assert "intent" not in PARAPHRASE_SYSTEM_PROMPT.lower().split("testing")[0]


class TestAugmentationFile:
    def _mini_dataset(self, tmp_path):
        path = tmp_path / "intents.json"
        path.write_text(json.dumps([
            {"utterance": "Cancel Bruno Silva's internet order.",
             "expected": "cancel_order"},
            {"utterance": "old paraphrase", "expected": "cancel_order",
             "source": "llm", "reviewed": False},
        ]))
        return path

    def test_appends_unreviewed_and_skips_llm_sources(self, tmp_path):
        path = self._mini_dataset(tmp_path)
        # ONE scripted response: only the author case may be paraphrased.
        # A second call (for the llm-sourced entry) would exhaust the script.
        llm = ScriptedLLM([{"paraphrases": [
            "Terminate the internet order of Bruno Silva.",
        ]}])
        stats = augment_dataset_file(path, llm, n=1, report=lambda _: None)
        assert stats == {"accepted": 1, "rejected": 0}

        raw = json.loads(path.read_text())
        assert len(raw) == 3
        added = raw[-1]
        assert added["expected"] == "cancel_order"   # label rides along
        assert added["source"] == "llm"
        assert added["reviewed"] is False

    def test_idempotent_on_utterance_text(self, tmp_path):
        path = self._mini_dataset(tmp_path)
        script = [{"paraphrases": ["Terminate the internet order of Bruno Silva."]}]
        augment_dataset_file(path, ScriptedLLM(list(script)),
                             n=1, report=lambda _: None)
        stats = augment_dataset_file(path, ScriptedLLM(list(script)),
                                     n=1, report=lambda _: None)
        assert stats["accepted"] == 0
        assert len(json.loads(path.read_text())) == 3

    def test_loader_review_gate(self, tmp_path):
        path = self._mini_dataset(tmp_path)
        assert len(load_intent_dataset(path)) == 1          # author only
        assert len(load_intent_dataset(path, include_unreviewed=True)) == 2
        # Flip the flag -> included.
        raw = json.loads(path.read_text())
        raw[1]["reviewed"] = True
        path.write_text(json.dumps(raw))
        assert len(load_intent_dataset(path)) == 2


def test_shipped_dataset_still_loads_with_defaults():
    # The shipped file may legitimately contain reviewed LLM paraphrases.
    # The invariant is that the default load NEVER returns unreviewed cases.
    cases = load_intent_dataset(FIXTURE)
    assert len(cases) >= 18
    assert all(c.reviewed for c in cases)
    assert sum(c.source == "author" for c in cases) >= 18
