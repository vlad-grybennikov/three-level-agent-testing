"""The three level evaluators (§3.2–§3.4). Everything below is computed from
a captured trace (Levels 1–2) or from frozen fixtures (Level 3). Nothing
here calls a judge model.

Level 1 (outcome):    projected final DB state vs G, verdict + diffs.
Level 2 (path):       node F1, dependency-edge F1 (value-matched), order
                      conformance, redundancy ratio. P as hard assertions.
Level 3 (components): each stage on frozen inputs. Accuracy / slot match /
                      Recall@k over the exposed candidate ranking.
"""

from __future__ import annotations

import json
from math import comb
from typing import Optional

from pydantic import BaseModel, Field

from ..pipeline import (
    BindingError,
    EntityExtractor,
    IntentClassifier,
    SelectionInput,
    SlotSet,
    ToolSelector,
    load_binder_class,
)
from .annotations import (
    DependencyGraph,
    GoalState,
    IntentCase,
    PolicyAssertion,
    Projection,
    TaskAnnotation,
)

# ---------------------------------------------------------------------------
# Level 1: end-to-end outcome
# ---------------------------------------------------------------------------


class Level1Result(BaseModel):
    passed: bool
    diffs: list[str] = Field(default_factory=list)


def evaluate_level1(trace: dict, goal: GoalState,
                    projection: Projection) -> Level1Result:
    """Projected final state must equal projected initial state + G patches."""
    initial = projection.apply(trace["initial_snapshot"])
    actual = projection.apply(trace["final_snapshot"])

    expected = {t: {rid: dict(row) for rid, row in rows.items()}
                for t, rows in initial.items()}
    diffs: list[str] = []
    for table, rows in goal.expected.items():
        for row_id, patch in rows.items():
            if table not in expected or row_id not in expected[table]:
                diffs.append(f"goal references unknown row {table}/{row_id}")
                continue
            expected[table][row_id].update(patch)

    for table, exp_rows in expected.items():
        act_rows = actual.get(table, {})
        for row_id, exp_row in exp_rows.items():
            if row_id not in act_rows:
                diffs.append(f"{table}/{row_id}: row missing in final state")
                continue
            for col, exp_val in exp_row.items():
                got = act_rows[row_id].get(col)
                if got != exp_val:
                    diffs.append(
                        f"{table}/{row_id}.{col}: expected {exp_val!r}, got {got!r}"
                    )
        for row_id in act_rows:
            if row_id not in exp_rows:
                diffs.append(f"{table}/{row_id}: unexpected new row")

    return Level1Result(passed=not diffs, diffs=diffs)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """τ-bench pass^k estimator: C(c, k) / C(n, k) for c successes in n runs."""
    if k > n:
        raise ValueError("k cannot exceed the number of runs")
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


# ---------------------------------------------------------------------------
# Level 2: integration (path)
# ---------------------------------------------------------------------------


class PolicyResult(BaseModel):
    assertion: PolicyAssertion
    passed: bool
    detail: str


class Level2Result(BaseModel):
    node_precision: float
    node_recall: float
    node_f1: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    order_conformance: float
    redundancy_ratio: float
    observed_ops: list[str]
    missing_ops: list[str]
    policy_results: list[PolicyResult]

    @property
    def policies_passed(self) -> bool:
        return all(p.passed for p in self.policy_results)


def _successful_calls(trace: dict) -> list[dict]:
    return [
        s for s in trace["steps"]
        if s.get("call") and s["call"]["tool"] != "finish"
        and s.get("error") is None
    ]


def _scalars(obj) -> set:
    """Matchable values: strings (len>=3) and non-trivial ints. Known
    limitation of edge inference by value-matching: coincidental
    equality of common values can produce spurious edges, and flows through
    values the tools never echo are invisible."""
    out: set = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str) and len(x) >= 3:
            out.add(x)
        elif isinstance(x, int) and not isinstance(x, bool) and abs(x) >= 10:
            out.add(x)

    walk(obj)
    return out


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def evaluate_level2(trace: dict, dep: DependencyGraph,
                    policies: list[PolicyAssertion]) -> Level2Result:
    calls = _successful_calls(trace)
    observed_tools = [c["call"]["tool"] for c in calls]
    observed_set = set(observed_tools)

    # -- nodes: required ops (tool + optional args subset) vs observed
    def op_matched(spec) -> bool:
        for c in calls:
            if c["call"]["tool"] != spec.tool:
                continue
            args = c["call"]["args"]
            if all(args.get(k) == v for k, v in spec.args_include.items()):
                return True
        return False

    required = dep.required_ops
    matched = [spec for spec in required if op_matched(spec)]
    required_tools = {spec.tool for spec in required}
    node_recall = len(matched) / len(required) if required else 1.0
    node_precision = (
        len(observed_set & required_tools) / len(observed_set)
        if observed_set else 1.0
    )

    # -- edges: infer observed data-flow edges by value-matching with
    #    EARLIEST-PRODUCER attribution: an argument value is credited to the
    #    first call whose result contained it. This keeps inference
    #    deterministic, but conflates re-derived values with their first
    #    derivation and misses values the agent transformed, a known
    #    limitation of trace-level edge recovery.
    inferred: set[tuple[str, str]] = set()
    producer: dict = {}  # scalar value -> tool that first produced it
    for c in calls:
        tool = c["call"]["tool"]
        for value in _scalars(c["call"]["args"]):
            if value in producer:
                inferred.add((producer[value], tool))
        for value in _scalars(c.get("result")):
            producer.setdefault(value, tool)
    declared = {(e.source, e.target) for e in dep.edges}
    edge_recall = len(inferred & declared) / len(declared) if declared else 1.0
    edge_precision = len(inferred & declared) / len(inferred) if inferred else 1.0

    # -- ordering: declared strict partial order, first-occurrence indices
    def first_index(tool: str) -> Optional[int]:
        for idx, c in enumerate(calls):
            if c["call"]["tool"] == tool:
                return idx
        return None

    satisfied = 0
    for pair in dep.ordering:
        a, b = first_index(pair.source), first_index(pair.target)
        if a is not None and b is not None and a < b:
            satisfied += 1
    order_conformance = satisfied / len(dep.ordering) if dep.ordering else 1.0

    # -- redundancy: identical successful (tool, args) repeats
    seen: set[str] = set()
    duplicates = 0
    for c in calls:
        key = c["call"]["tool"] + json.dumps(c["call"]["args"], sort_keys=True)
        if key in seen:
            duplicates += 1
        seen.add(key)
    redundancy_ratio = round(duplicates / len(calls), 4) if calls else 0.0

    # -- P: hard assertions over the successful-call sequence
    policy_results = []
    for assertion in policies:
        if assertion.kind == "require_before":
            then_idxs = [i for i, c in enumerate(calls)
                         if c["call"]["tool"] == assertion.then]
            first_idxs = [i for i, c in enumerate(calls)
                          if c["call"]["tool"] == assertion.first]
            bad = [i for i in then_idxs
                   if not any(j < i for j in first_idxs)]
            passed = not bad
            detail = ("ok" if passed else
                      f"{assertion.then} executed without prior {assertion.first}")
        else:  # forbid_call
            passed = assertion.tool not in observed_set
            detail = "ok" if passed else f"{assertion.tool} was executed"
        policy_results.append(PolicyResult(
            assertion=assertion, passed=passed, detail=detail))

    return Level2Result(
        node_precision=round(node_precision, 4),
        node_recall=round(node_recall, 4),
        node_f1=_f1(node_precision, node_recall),
        edge_precision=round(edge_precision, 4),
        edge_recall=round(edge_recall, 4),
        edge_f1=_f1(edge_precision, edge_recall),
        order_conformance=round(order_conformance, 4),
        redundancy_ratio=redundancy_ratio,
        observed_ops=observed_tools,
        missing_ops=[s.tool for s in required if s not in matched],
        policy_results=policy_results,
    )


# ---------------------------------------------------------------------------
# Level 3: components on frozen inputs
# ---------------------------------------------------------------------------


class Level3Result(BaseModel):
    intent_correct: bool
    predicted_intent: str
    slot_results: dict[str, bool] = Field(default_factory=dict)
    slot_accuracy: float = 1.0
    selection_hits: list[bool] = Field(default_factory=list)
    selection_recall_at_k: float = 1.0
    binding_valid: list[bool] = Field(default_factory=list)
    binding_exact: list[bool] = Field(default_factory=list)
    binding_accuracy: float = 1.0


def _slot_match(got, expected) -> bool:
    if got is None:
        return False
    if isinstance(expected, str) and isinstance(got, str):
        # Separator-insensitive: "Fiber 500" ~ "fiber-500" ~ "fiber500".
        a = "".join(ch for ch in got.lower() if ch.isalnum())
        b = "".join(ch for ch in expected.lower() if ch.isalnum())
        return a == b or a in b or b in a
    return got == expected


def evaluate_level3(task: TaskAnnotation, config, llm) -> Level3Result:
    """Each component in isolation: no loop, no database mutation.

    Intent and extraction run on the task instruction. Selection cases run on
    frozen SelectionInputs built from the annotated slots + fixture history.
    Recall@k is measured on the exposed candidate ranking, not just top-1.
    """
    fx = task.level3
    assert fx is not None, f"task {task.id} has no level3 fixtures"

    classifier = IntentClassifier(llm, config)
    predicted = classifier.run(task.instruction).intent
    intent_correct = predicted == fx.expected_intent

    extractor = EntityExtractor(llm, config)
    slots = extractor.run(task.instruction, fx.expected_intent)
    slot_results = {
        name: _slot_match(getattr(slots, name, None), expected)
        for name, expected in fx.expected_slots.items()
    }
    slot_accuracy = (
        round(sum(slot_results.values()) / len(slot_results), 4)
        if slot_results else 1.0
    )

    selector = ToolSelector(llm, config)
    frozen_slots = SlotSet.model_validate({
        k: v for k, v in fx.expected_slots.items()
        if k in SlotSet.model_fields
    })
    hits = []
    for case in fx.selection_cases:
        sel = selector.run(SelectionInput(
            intent=fx.expected_intent, slots=frozen_slots,
            history=case.history,
        ))
        hits.append(case.expected in [c.tool for c in sel.candidates])
    recall_at_k = round(sum(hits) / len(hits), 4) if hits else 1.0

    # Stage 4 through the CONFIGURED binder: an injected defective binder
    # is measured, not bypassed. Judged on schema validity and exact
    # argument match against the annotation.
    binder = load_binder_class(config.binder_class)(llm, config)
    binding_valid, binding_exact = [], []
    for case in fx.binding_cases:
        sel_input = SelectionInput(
            intent=fx.expected_intent, slots=frozen_slots,
            history=case.history,
        )
        try:
            call = binder.run(case.tool, sel_input)
            binding_valid.append(True)
            binding_exact.append(call.args == case.expected_args)
        except BindingError:
            binding_valid.append(False)
            binding_exact.append(False)
    binding_accuracy = (
        round(sum(binding_exact) / len(binding_exact), 4)
        if binding_exact else 1.0
    )

    return Level3Result(
        intent_correct=intent_correct,
        predicted_intent=predicted,
        slot_results=slot_results,
        slot_accuracy=slot_accuracy,
        selection_hits=hits,
        selection_recall_at_k=recall_at_k,
        binding_valid=binding_valid,
        binding_exact=binding_exact,
        binding_accuracy=binding_accuracy,
    )


# -- intent classification as a dataset metric (per-class F1) ---------------


class IntentClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class IntentF1Report(BaseModel):
    accuracy: float
    macro_f1: float
    per_class: dict[str, IntentClassMetrics]
    errors: list[dict] = Field(default_factory=list)


def evaluate_intent_classification(cases: list[IntentCase], config,
                                   llm) -> IntentF1Report:
    """Stage 1 over a labeled utterance set: per-class P/R/F1 + macro F1.

    Metrics are computed over the label set present in the dataset. A
    prediction outside that set counts as a miss for its true class only.
    """
    classifier = IntentClassifier(llm, config)
    preds = [(case, classifier.run(case.utterance).intent) for case in cases]
    labels = sorted({case.expected for case in cases})

    per_class: dict[str, IntentClassMetrics] = {}
    f1_values = []
    for label in labels:
        tp = sum(1 for c, p in preds if c.expected == label and p == label)
        fp = sum(1 for c, p in preds if c.expected != label and p == label)
        fn = sum(1 for c, p in preds if c.expected == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[label] = IntentClassMetrics(
            precision=round(precision, 4), recall=round(recall, 4),
            f1=round(f1, 4), support=tp + fn,
        )
        f1_values.append(f1)

    return IntentF1Report(
        accuracy=round(
            sum(1 for c, p in preds if p == c.expected) / len(preds), 4
        ),
        macro_f1=round(sum(f1_values) / len(f1_values), 4),
        per_class=per_class,
        errors=[
            {"utterance": c.utterance, "expected": c.expected, "predicted": p}
            for c, p in preds if p != c.expected
        ],
    )
