"""Harness runner: episodes -> traces -> per-level verdicts -> report.

    python -m telecom_aut.testing.runner approach_tests/tasks --k 3
        [--config cfg.json] [--out report.json]

Per task: k live episodes (fresh seed each, clean single-turn episodes),
Level 1 + Level 2 evaluated per trace, Level 3 evaluated once on frozen
fixtures, pass^k from Level-1 verdicts. Offline (scripted-LLM) use goes
through evaluate_trace()/evaluate_level3 directly (see tests/test_harness.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..agent import TelecomAgent
from ..config import AgentConfig
from ..environment import TelecomEnv
from .annotations import TaskAnnotation, load_tasks
from .levels import (
    Level1Result,
    Level2Result,
    evaluate_level1,
    evaluate_level2,
    evaluate_level3,
    pass_hat_k,
)


def evaluate_trace(trace: dict, task: TaskAnnotation) -> dict:
    """Levels 1 and 2 for one captured trace."""
    l1: Level1Result = evaluate_level1(trace, task.goal, task.projection)
    l2: Level2Result = evaluate_level2(trace, task.dependency, task.policies)
    return {
        "level1": l1.model_dump(),
        "level2": l2.model_dump(),
        "level1_passed": l1.passed,
        "policies_passed": l2.policies_passed,
    }


def run_task(task: TaskAnnotation, config: AgentConfig, k: int,
             llm=None, judge_llm=None, use_variants: bool = False) -> dict:
    """k episodes on fresh environments + one Level-3 pass.

    judge_llm, if given, grades each episode's success from the trajectory
    (classic LLM-as-judge, see judge.py). Judge verdicts aggregate into
    their own judge_pass_hat_k column, parallel to (never replacing)
    Level 1's deterministic pass^k, so judge-vs-oracle agreement is
    measurable. use_variants rotates episodes over the canonical
    instruction plus REVIEWED variants (same G/D/P).
    """
    from .judge import judge_episode_outcome

    phrasings = task.phrasings() if use_variants else [task.instruction]
    episodes = []
    successes = 0
    judge_successes = 0
    for i in range(k):
        instruction = phrasings[i % len(phrasings)]
        env = TelecomEnv.fresh()
        agent = TelecomAgent(env, config, llm=llm)
        try:
            result = agent.run_episode(instruction)
        except Exception as exc:  # model/auth/network death mid-episode
            env.close()
            episodes.append({
                "run": i,
                "instruction": instruction,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "level1_passed": False,
                "policies_passed": False,
            })
            continue  # an errored episode counts as a failure, not a crash
        verdicts = evaluate_trace(result.trace, task)
        env.close()
        judge = (judge_episode_outcome(result.trace, judge_llm).model_dump()
                 if judge_llm is not None else None)
        if judge is not None and judge["passed"]:
            judge_successes += 1
        episodes.append({
            "run": i,
            "instruction": instruction,
            "status": result.status,
            "summary": result.summary,
            **verdicts,
            "judge": judge,
            "trace": result.trace,
        })
        successes += verdicts["level1_passed"]

    env = TelecomEnv.fresh()
    agent = TelecomAgent(env, config, llm=llm)
    level3 = (evaluate_level3(task, config, agent.llm).model_dump()
              if task.level3 else None)
    env.close()

    return {
        "task_id": task.id,
        "capability": task.capability,
        "k": k,
        "level1_successes": successes,
        "pass_hat_k": {
            str(kk): round(pass_hat_k(k, successes, kk), 4)
            for kk in range(1, k + 1)
        },
        "judge_successes": judge_successes if judge_llm is not None else None,
        "judge_pass_hat_k": {
            str(kk): round(pass_hat_k(k, judge_successes, kk), 4)
            for kk in range(1, k + 1)
        } if judge_llm is not None else None,
        "level3": level3,
        "episodes": episodes,
    }


def main(argv: list[str] | None = None) -> int:
    from ..envfile import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="telecom_aut.testing.runner")
    parser.add_argument("tasks", help="directory of task JSON files")
    parser.add_argument("--k", type=int, default=1,
                        help="episodes per task (pass^k needs k > 1)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="report.json")
    parser.add_argument("--judge", action="store_true",
                        help="grade each episode with an LLM judge"
                             " (own judge_pass_hat_k column)")
    parser.add_argument("--judge-model", default=None,
                        help="model tag for the judge (default: agent model)")
    parser.add_argument("--variants", action="store_true",
                        help="rotate episodes over reviewed phrasing variants")
    parser.add_argument("--intent-dataset", default=None,
                        help="labeled utterances file -> per-class intent F1"
                             " (Level 3) appended to the report")
    args = parser.parse_args(argv)

    config = AgentConfig.from_file(args.config) if args.config else AgentConfig()

    judge_llm = None
    if args.judge:
        from ..llm import build_chat_client
        judge_cfg = config.model.model_copy(deep=True)
        if args.judge_model:
            judge_cfg.model = args.judge_model
        judge_llm = build_chat_client(judge_cfg)
    tasks = load_tasks(args.tasks)
    print(f"{len(tasks)} tasks, k={args.k}, config={config.config_hash()}")

    report = {"config_hash": config.config_hash(),
              "model_id": config.model.model, "k": args.k, "tasks": []}
    for task in tasks:
        print(f"\n=== {task.id} ({task.capability}) ===")
        outcome = run_task(task, config, args.k, judge_llm=judge_llm,
                           use_variants=args.variants)
        report["tasks"].append(outcome)
        for ep in outcome["episodes"]:
            if ep["status"] == "error":
                print(f"  run {ep['run']}: ERROR: {ep['error']}")
                continue
            l2 = ep["level2"]
            print(f"  run {ep['run']}: L1 "
                  f"{'PASS' if ep['level1_passed'] else 'FAIL'}"
                  f"  node_f1={l2['node_f1']}"
                  f"  edge_f1={l2['edge_f1']}"
                  f"  order={l2['order_conformance']}"
                  f"  policies={'ok' if ep['policies_passed'] else 'VIOLATED'}")
            for diff in ep["level1"]["diffs"][:4]:
                print(f"      L1 diff: {diff}")
            if ep.get("judge"):
                j = ep["judge"]
                print(f"      judge: {'PASS' if j['passed'] else 'FAIL'}"
                      + (f": {j['reasoning'][:100]}" if j["reasoning"] else ""))
        if outcome["level3"]:
            l3 = outcome["level3"]
            print(f"  L3: intent={'ok' if l3['intent_correct'] else 'WRONG'}"
                  f"  slots={l3['slot_accuracy']}"
                  f"  recall@k={l3['selection_recall_at_k']}")
        print(f"  pass^k (L1): {outcome['pass_hat_k']}")
        if outcome.get("judge_pass_hat_k") is not None:
            print(f"  pass^k (judge): {outcome['judge_pass_hat_k']}")

    if args.intent_dataset:
        from ..llm import build_chat_client
        from .annotations import load_intent_dataset
        from .levels import evaluate_intent_classification

        cases = load_intent_dataset(args.intent_dataset)
        print(f"\n=== intent classification ({len(cases)} reviewed cases) ===")
        f1_report = evaluate_intent_classification(
            cases, config, build_chat_client(config.model)
        )
        report["intent_classification"] = f1_report.model_dump()
        print(f"  accuracy={f1_report.accuracy}  macro_f1={f1_report.macro_f1}")
        for label, metrics in sorted(f1_report.per_class.items()):
            print(f"  {label:24s} P={metrics.precision}  R={metrics.recall}"
                  f"  F1={metrics.f1}  n={metrics.support}")
        for error in f1_report.errors:
            print(f"  MISS: {error['utterance']!r} "
                  f"expected={error['expected']} got={error['predicted']}")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
