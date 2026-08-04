"""LLM-generated instruction variants (task-set augmentation).

    python -m telecom_aut.testing.variants approach_tests/tasks --n 4
        [--config cfg.json] [--include TASK_ID]

Pipeline per task:
  1. an LLM paraphrases the operator request N ways,
  2. a DETERMINISTIC entity guard rejects any variant that lost a critical
     literal (subscriber name, prefixed ids, numeric tokens like dates and
     plan speeds), the classic semantic-drift failure of paraphrasing,
  3. survivors are appended to the task JSON as reviewed=false.

Nothing consumes an unreviewed variant: the runner uses task.phrasings(),
which excludes them until a human flips the flag. Variants never touch
(G, D, P): every phrasing is graded against the same annotation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .annotations import InstructionVariant, TaskAnnotation, load_tasks

VARIANT_SYSTEM_PROMPT = """\
You generate phrasing variants of internal care-ops requests for testing an
agent. Rules:
- Keep the SAME meaning, target, and outcome — only the wording changes.
- Keep the operator perspective (third person about the subscriber).
- Keep every name, id, plan, date, and time window from the original.
- Vary structure and register (terse ticket style, polite, verbose).
Respond with only JSON: {"variants": ["...", "..."]}."""

VARIANT_USER_TEMPLATE = """\
Original request: {instruction}

Produce {n} distinct phrasing variants."""

_ID_RE = re.compile(r"\b(?:SUB|ORD|APT|SLT|INV)-[0-9A-F]{4}\b")
_NUM_RE = re.compile(r"\d+(?::\d+)*")  # plain numbers and clock times


def _norm(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def critical_literals(task: TaskAnnotation) -> dict[str, list[str]]:
    """What every variant must preserve, derived deterministically.

    names: the subscriber's full name (from the Level-3 annotation when
    present). ids: every prefixed entity id in the instruction. numbers:
    every numeric token (dates, speeds) in the instruction.
    """
    names = []
    if task.level3:
        name = task.level3.expected_slots.get("subscriber_name")
        if isinstance(name, str):
            names.append(name)
    return {
        "names": names,
        "ids": _ID_RE.findall(task.instruction),
        "numbers": _NUM_RE.findall(task.instruction),
    }


def preserves_entities(variant: str, required: dict[str, list[str]]) -> bool:
    normed = _norm(variant)
    for name in required["names"]:
        if _norm(name) not in normed:
            return False
    for entity_id in required["ids"]:
        if entity_id not in variant:
            return False
    for number in required["numbers"]:
        if number not in variant:
            return False
    return True


def generate_variants(task: TaskAnnotation, llm,
                      n: int = 4) -> tuple[list[str], list[str]]:
    """-> (accepted, rejected). Accepted still require human review."""
    data = llm.complete_json(
        VARIANT_SYSTEM_PROMPT,
        VARIANT_USER_TEMPLATE.format(instruction=task.instruction, n=n),
    )
    raw = data.get("variants") if isinstance(data, dict) else None
    candidates = [v.strip() for v in raw if isinstance(v, str)] \
        if isinstance(raw, list) else []
    required = critical_literals(task)
    accepted, rejected = [], []
    for candidate in candidates:
        if candidate and candidate != task.instruction \
                and preserves_entities(candidate, required):
            accepted.append(candidate)
        else:
            rejected.append(candidate)
    return accepted, rejected


def append_variants_to_file(task_path: str | Path, texts: list[str]) -> None:
    """Persist accepted variants as reviewed=false (idempotent on text)."""
    path = Path(task_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    existing = {v["text"] for v in data.get("variants", [])}
    data.setdefault("variants", []).extend(
        InstructionVariant(text=t, source="llm", reviewed=False).model_dump()
        for t in texts if t not in existing
    )
    TaskAnnotation.model_validate(data)  # never write an invalid task file
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    from ..config import AgentConfig
    from ..llm import build_chat_client

    from ..envfile import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="telecom_aut.testing.variants")
    parser.add_argument("tasks", help="directory of task JSON files")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--config", default=None)
    parser.add_argument("--include", action="append", default=None,
                        help="only these task ids (repeatable)")
    args = parser.parse_args(argv)

    config = AgentConfig.from_file(args.config) if args.config else AgentConfig()
    llm = build_chat_client(config.model)

    for path in sorted(Path(args.tasks).glob("*.json")):
        task = TaskAnnotation.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
        if args.include and task.id not in args.include:
            continue
        accepted, rejected = generate_variants(task, llm, args.n)
        append_variants_to_file(path, accepted)
        print(f"{task.id}: +{len(accepted)} variants "
              f"({len(rejected)} rejected by entity guard)")
        for text in accepted:
            print(f"   [unreviewed] {text}")
        for text in rejected:
            print(f"   [REJECTED]   {text}")
    print("\nReview: set \"reviewed\": true on acceptable variants; the "
          "runner ignores the rest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
