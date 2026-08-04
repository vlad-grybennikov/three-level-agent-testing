"""Task annotations: the (G, D, P) triple plus Level-3 fixtures (paper §3.1).

Every field is computable-against from a captured trace:
  G:  expected final state as cell-level patches over the projected initial
      snapshot, with an everything-else-unchanged rule (Level 1).
  D:  required operations, data-dependency edges, and ordering constraints
      as a strict partial order (Level 2).
  P:  hard policy assertions over the successful-call sequence (Level 2).
  level3: frozen inputs + expected outputs per decision component.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Projection(BaseModel):
    """What Level 1 discards before comparing states (§3.2).

    Defaults: the events audit log (its content is path-dependent: Level 2's
    subject, not Level 1's) and runtime-written timestamps (they encode step
    counts, not outcomes).
    """

    model_config = ConfigDict(extra="forbid")

    drop_tables: list[str] = Field(default_factory=lambda: ["events"])
    drop_fields: dict[str, list[str]] = Field(
        default_factory=lambda: {"orders": ["cancelled_at"]}
    )

    def apply(self, snapshot: dict) -> dict[str, dict]:
        """snapshot -> {table: {pk: row_without_dropped_fields}}."""
        out: dict[str, dict] = {}
        for table, rows in snapshot["tables"].items():
            if table in self.drop_tables:
                continue
            dropped = set(self.drop_fields.get(table, []))
            pk = "seq" if table == "events" else "id"
            out[table] = {
                str(row[pk]): {k: v for k, v in row.items() if k not in dropped}
                for row in rows
            }
        return out


DEFAULT_PROJECTION = Projection()


class GoalState(BaseModel):
    """G: cell-level patches over the projected initial state.

    expected[table][row_id][column] = value. Rows/columns not mentioned must
    equal the initial state (the everything-else-unchanged rule), so an
    empty `expected` annotates a read-only task.
    """

    model_config = ConfigDict(extra="forbid")

    expected: dict[str, dict[str, dict]] = Field(default_factory=dict)


class OpSpec(BaseModel):
    """A required operation: tool name, optionally with an argument subset."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    args_include: dict = Field(default_factory=dict)


class Edge(BaseModel):
    """source -> target between tool names (first-occurrence semantics)."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class DependencyGraph(BaseModel):
    """D: required ops + data-dependency edges + ordering partial order."""

    model_config = ConfigDict(extra="forbid")

    required_ops: list[OpSpec] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)       # data flows source->target
    ordering: list[Edge] = Field(default_factory=list)    # source strictly before target


class PolicyAssertion(BaseModel):
    """P: a hard pass/fail assertion over the successful-call sequence.

    kinds:
      require_before: every successful `then` call must be preceded by at
                      least one successful `first` call.
      forbid_call:    `tool` must never execute successfully.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["require_before", "forbid_call"]
    first: Optional[str] = None
    then: Optional[str] = None
    tool: Optional[str] = None


class SelectionCase(BaseModel):
    """Frozen stage-3 input: history lines + the expected next tool."""

    model_config = ConfigDict(extra="forbid")

    history: list[str] = Field(default_factory=list)
    expected: str


class BindingCase(BaseModel):
    """Frozen stage-4 input: tool + history, judged on schema validity and
    exact argument match."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    history: list[str] = Field(default_factory=list)
    expected_args: dict = Field(default_factory=dict)


class IntentCase(BaseModel):
    """One labeled utterance for the intent-classification F1 dataset.

    Hand-written entries default to source="author"/reviewed=True.
    LLM-paraphrased entries are appended as source="llm"/reviewed=False and
    stay out of evaluations until a human flips the flag.
    """

    model_config = ConfigDict(extra="forbid")

    utterance: str
    expected: str
    source: Literal["author", "llm"] = "author"
    reviewed: bool = True


class Level3Fixtures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_intent: str
    # Slot name -> expected value. Strings match case-insensitively and by
    # containment either way ("Fiber 500" ~ "fiber-500").
    expected_slots: dict[str, object] = Field(default_factory=dict)
    selection_cases: list[SelectionCase] = Field(default_factory=list)
    binding_cases: list[BindingCase] = Field(default_factory=list)


class AnnotationMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    annotator: str = "author"
    minutes: Optional[int] = None  # annotation time, for cost reporting


class InstructionVariant(BaseModel):
    """An alternative phrasing of the same task (identical G, D, P).

    LLM-generated variants start reviewed=False and are excluded from runs
    until a human flips the flag. Paraphrases of one instruction are not
    independent samples: they widen phrasing coverage without growing the
    effective task count.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    source: Literal["author", "llm"] = "llm"
    reviewed: bool = False


class TaskAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    capability: str                  # one of the five intents
    instruction: str
    variants: list[InstructionVariant] = Field(default_factory=list)
    goal: GoalState = Field(default_factory=GoalState)
    dependency: DependencyGraph = Field(default_factory=DependencyGraph)
    policies: list[PolicyAssertion] = Field(default_factory=list)
    level3: Optional[Level3Fixtures] = None
    projection: Projection = Field(default_factory=lambda: DEFAULT_PROJECTION.model_copy(deep=True))
    annotation: AnnotationMeta = Field(default_factory=AnnotationMeta)

    def phrasings(self, include_unreviewed: bool = False) -> list[str]:
        """Canonical instruction + approved variants, in stable order."""
        return [self.instruction] + [
            v.text for v in self.variants if v.reviewed or include_unreviewed
        ]


def load_task(path: str | Path) -> TaskAnnotation:
    return TaskAnnotation.model_validate(
        json.loads(Path(path).read_text(encoding="utf-8"))
    )


def load_tasks(directory: str | Path) -> list[TaskAnnotation]:
    return [load_task(p) for p in sorted(Path(directory).glob("*.json"))]


def load_intent_dataset(path: str | Path,
                        include_unreviewed: bool = False) -> list[IntentCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [IntentCase.model_validate(item) for item in raw]
    return [c for c in cases if c.reviewed or include_unreviewed]
