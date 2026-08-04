"""LLM paraphrase enrichment for the intent-classification dataset.

    python -m telecom_aut.testing.intent_augment \
        approach_tests/fixtures/level3_intents.json --n 3 [--config cfg.json]

Per author-written case:
  1. the LLM paraphrases the operator request N ways (label rides along
     unchanged: the LLM never assigns labels, only rewords),
  2. a DETERMINISTIC entity guard rejects paraphrases that lost a critical
     literal (person names, prefixed ids, numeric tokens), the usual
     semantic-drift failure of paraphrasing,
  3. survivors are appended to the same file as source="llm",
     reviewed=false.

Nothing consumes an unreviewed case: load_intent_dataset() filters them out
until a human flips the flag. Only source="author" cases are paraphrased,
never paraphrases of paraphrases. Re-running is idempotent on utterance
text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .annotations import IntentCase
from .variants import _NUM_RE, preserves_entities

PARAPHRASE_SYSTEM_PROMPT = """\
You generate phrasing variants of internal care-ops operator requests for
testing an intent classifier. Rules:
- Keep the SAME meaning and the same underlying request type — only the
  wording changes. Never turn one kind of request into another.
- Keep the operator perspective (third person about the subscriber).
- Keep every person name, id, plan, number, date, and time from the
  original.
- Vary structure and register (terse ticket style, polite, verbose,
  colloquial).
Respond with only JSON: {"paraphrases": ["...", "..."]}."""

PARAPHRASE_USER_TEMPLATE = """\
Original operator request: {utterance}

Produce {n} distinct paraphrases."""

_ID_RE = re.compile(r"\b(?:SUB|ORD|APT|SLT|INV)-[0-9A-F]{4}\b")
_CAP_RUN_RE = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")


def _person_names(utterance: str) -> list[str]:
    """Maximal capitalized-word runs, minus a sentence-initial verb.

    "Cancel Bruno Silva's order" -> run "Cancel Bruno Silva" starts the
    sentence with 3+ words, so the leading verb is dropped -> "Bruno Silva".
    A sentence-initial TWO-word run ("Vlad Grybennikov asked...") is kept
    whole. Operator requests always use full names, so a bare
    verb+firstname opener doesn't occur in this corpus.
    """
    names = []
    for match in _CAP_RUN_RE.finditer(utterance):
        words = match.group().split()
        if match.start() == 0 and len(words) >= 3:
            words = words[1:]
        names.append(" ".join(words))
    return names


def critical_literals_for_utterance(utterance: str) -> dict[str, list[str]]:
    """Guard inputs derived from the utterance text alone: person names,
    prefixed entity ids, and numeric tokens."""
    return {
        "names": _person_names(utterance),
        "ids": _ID_RE.findall(utterance),
        "numbers": _NUM_RE.findall(utterance),
    }


def generate_intent_paraphrases(case: IntentCase, llm,
                                n: int = 3) -> tuple[list[str], list[str]]:
    """-> (accepted, rejected). Accepted still require human review."""
    data = llm.complete_json(
        PARAPHRASE_SYSTEM_PROMPT,
        PARAPHRASE_USER_TEMPLATE.format(utterance=case.utterance, n=n),
    )
    raw = data.get("paraphrases") if isinstance(data, dict) else None
    candidates = [p.strip() for p in raw if isinstance(p, str)] \
        if isinstance(raw, list) else []
    required = critical_literals_for_utterance(case.utterance)
    accepted, rejected = [], []
    for candidate in candidates:
        if candidate and candidate != case.utterance \
                and preserves_entities(candidate, required):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    return accepted, rejected


def augment_dataset_file(path: str | Path, llm, n: int = 3,
                         report=print) -> dict:
    """Paraphrase every author case in the file and append survivors in place.

    Returns {"accepted": int, "rejected": int}. Idempotent: an utterance
    already present (any source) is never added twice.
    """
    file_path = Path(path)
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    cases = [IntentCase.model_validate(item) for item in raw]
    existing = {c.utterance for c in cases}

    stats = {"accepted": 0, "rejected": 0}
    additions: list[IntentCase] = []
    for case in cases:
        if case.source != "author":
            continue  # never paraphrase paraphrases
        accepted, rejected = generate_intent_paraphrases(case, llm, n)
        stats["rejected"] += len(rejected)
        for text in rejected:
            report(f"  [REJECTED]   ({case.expected}) {text}")
        for text in accepted:
            if text in existing:
                continue
            existing.add(text)
            stats["accepted"] += 1
            additions.append(IntentCase(
                utterance=text, expected=case.expected,
                source="llm", reviewed=False,
            ))
            report(f"  [unreviewed] ({case.expected}) {text}")

    file_path.write_text(
        json.dumps([c.model_dump() for c in cases + additions], indent=2)
        + "\n",
        encoding="utf-8",
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    from ..config import AgentConfig
    from ..llm import build_chat_client

    from ..envfile import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="telecom_aut.testing.intent_augment")
    parser.add_argument("dataset", help="intent dataset JSON file")
    parser.add_argument("--n", type=int, default=3,
                        help="paraphrases requested per author case")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = AgentConfig.from_file(args.config) if args.config else AgentConfig()
    llm = build_chat_client(config.model)

    stats = augment_dataset_file(args.dataset, llm, args.n)
    print(f"\n+{stats['accepted']} paraphrases appended "
          f"({stats['rejected']} rejected by the entity guard).")
    print("Review: set \"reviewed\": true on acceptable entries; "
          "load_intent_dataset() ignores the rest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
