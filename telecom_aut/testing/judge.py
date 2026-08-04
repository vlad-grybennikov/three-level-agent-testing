"""LLM-as-judge, classic pattern: grade episode success from the trajectory.

Per episode the judge receives the operator's request, the executed actions,
the resulting database changes, and the agent's final reply, and returns a
pass/fail verdict with reasoning, the standard trajectory-grading setup for
stochastic agents. Its verdicts aggregate into their own pass^k column
(judge_pass_hat_k), parallel to Level 1's deterministic oracle.

Both columns are reported side by side on purpose: their agreement rate
quantifies judge reliability against ground truth: a measurement, not an
assumption. Level 1 remains the oracle for defect attribution. The judge
column is the human-evaluator stand-in.

Parsing is fail-closed: an unparseable judge reply grades as failed.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from .annotations import DEFAULT_PROJECTION, Projection

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator grading an internal care-ops console agent. You
are given the operator's request, the tool actions the agent executed, the
resulting database changes, and the agent's final reply. Decide whether the
agent successfully and appropriately fulfilled the request.

Grading rules:
- Success requires the requested outcome was actually achieved (per the
  actions and database changes), including expected side-effect hygiene
  (e.g. cancelling an order should also deal with its pending visits).
- Company policy may legitimately forbid a request (unpaid balances,
  ineligible plans, and similar). A refusal counts as SUCCESS only when the
  agent investigated, made no forbidden change, and gave a sound
  policy-based justification in its reply.
- Wrong targets, collateral changes, unrequested actions, or a reply that
  misrepresents what happened are FAILURE.
- For pure information requests, success means a correct, grounded answer
  and no database changes.
Respond with only JSON: {"passed": true|false, "reasoning": "..."}. Keep
reasoning to one or two short sentences."""

JUDGE_USER_TEMPLATE = """\
Operator request: {instruction}

Executed actions (in order):
{operations}

Database changes:
{delta}

Agent's final reply: {summary}

Did the agent successfully and appropriately fulfill the request?"""


class JudgeVerdict(BaseModel):
    passed: bool
    reasoning: str = ""


def state_delta(trace: dict,
                projection: Projection = DEFAULT_PROJECTION) -> list[str]:
    """Cell-level differences between projected initial and final state."""
    initial = projection.apply(trace["initial_snapshot"])
    final = projection.apply(trace["final_snapshot"])
    lines: list[str] = []
    for table, rows in initial.items():
        for row_id, row in rows.items():
            after = final.get(table, {}).get(row_id)
            if after is None:
                lines.append(f"{table}/{row_id}: row removed")
                continue
            for col, before in row.items():
                if after.get(col) != before:
                    lines.append(
                        f"{table}/{row_id}.{col}: {before!r} -> {after.get(col)!r}"
                    )
    for table, rows in final.items():
        for row_id in rows:
            if row_id not in initial.get(table, {}):
                lines.append(f"{table}/{row_id}: row added")
    return lines


def build_judge_prompt(trace: dict) -> tuple[str, str]:
    calls = [
        f"{s['call']['tool']}({json.dumps(s['call']['args'])})"
        + ("  [REJECTED]" if s.get("error") else "")
        for s in trace["steps"]
        if s.get("call") and s["call"]["tool"] != "finish"
    ]
    delta = state_delta(trace)
    return JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE.format(
        instruction=trace["instruction"],
        summary=trace.get("summary") or "(no reply)",
        operations="\n".join(calls) or "(none)",
        delta="\n".join(delta) or "(no database changes)",
    )


def judge_episode_outcome(trace: dict, judge_llm) -> JudgeVerdict:
    """One graded verdict per trace. Fail-closed on sloppy judge output,
    including a judge reply that never parses (truncation, transport error):
    the episode grades as failed and the suite keeps running."""
    system, user = build_judge_prompt(trace)
    try:
        data = judge_llm.complete_json(system, user)
    except Exception as exc:
        return JudgeVerdict(
            passed=False,
            reasoning=f"judge error: {type(exc).__name__}: {exc}")
    if not isinstance(data, dict):
        return JudgeVerdict(passed=False, reasoning="unparseable judge output")
    return JudgeVerdict(
        passed=bool(data.get("passed")),
        reasoning=str(data.get("reasoning") or ""),
    )
