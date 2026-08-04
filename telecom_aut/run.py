"""Run one episode from the command line.

    python -m telecom_aut.run "Reschedule Alice Nguyen's repair visit to
                               the afternoon of August 5." \
        [--db path.db] [--config config.json] [--model qwen3.5] [--json out.json]

Requires a running model endpoint (default: Ollama at localhost:11434 serving
the model named in config, `ollama pull qwen3.5` first). The database is
freshly seeded per run unless --db points at a file you want inspected
afterwards. Even then it is re-seeded at episode start (single-episode
runner).
"""

from __future__ import annotations

import argparse
import json
import sys

from .agent import TelecomAgent
from .config import AgentConfig
from .environment import TelecomEnv


def main(argv: list[str] | None = None) -> int:
    from .envfile import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="telecom_aut.run", description=__doc__)
    parser.add_argument("instruction", help="single-turn customer instruction")
    parser.add_argument("--db", default=":memory:",
                        help="SQLite path (default: in-memory)")
    parser.add_argument("--config", default=None,
                        help="AgentConfig JSON file (default: clean baseline)")
    parser.add_argument("--model", default=None,
                        help="override config.model.model (e.g. qwen3.5:14b)")
    parser.add_argument("--reasoning", default=None,
                        choices=["none", "low", "medium", "high"],
                        help="override config.model.reasoning_effort")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="write the EpisodeResult JSON here")
    args = parser.parse_args(argv)

    config = AgentConfig.from_file(args.config) if args.config else AgentConfig()
    if args.model or args.reasoning:
        config = config.model_copy(deep=True)
        if args.model:
            config.model.model = args.model
        if args.reasoning:
            config.model.reasoning_effort = args.reasoning

    env = TelecomEnv.fresh(args.db)
    agent = TelecomAgent(env, config)

    print(f"model={config.model.model}  config_hash={config.config_hash()}",
          flush=True)
    print(f"instruction: {args.instruction}\n", flush=True)

    # Live progress: print intent/slots/steps the moment each lands.
    printed = {"intent": False, "slots": False, "steps": 0}

    def on_state(state: dict) -> None:
        if not printed["intent"] and state.get("intent"):
            print(f"intent: {state['intent']}", flush=True)
            printed["intent"] = True
        if not printed["slots"] and state.get("slots") is not None:
            filled = {k: v for k, v in state["slots"].items() if v is not None}
            print(f"slots:  {json.dumps(filled)}\n", flush=True)
            printed["slots"] = True
        steps = state.get("steps") or []
        while printed["steps"] < len(steps):
            step = steps[printed["steps"]]
            ranked = " > ".join(
                c["tool"]
                for c in (step.get("selection") or {}).get("candidates", [])
            )
            print(f"step {step['index']:2d}  candidates: {ranked}", flush=True)
            if step.get("call"):
                print(f"         call: {step['call']['tool']}"
                      f"({json.dumps(step['call']['args'])})", flush=True)
            if step.get("error"):
                print(f"         error: {step['error']}", flush=True)
            elif (step.get("result") is not None
                  and step["call"]["tool"] != "finish"):
                out = json.dumps(step["result"], default=str)
                print(f"         result: {out[:200]}"
                      f"{'...' if len(out) > 200 else ''}", flush=True)
            printed["steps"] += 1

    result = agent.run_episode(args.instruction, on_state=on_state)

    print(f"\nstatus: {result.status}")
    if result.summary:
        print(f"summary: {result.summary}")

    events = env.snapshot()["tables"]["events"]
    print(f"\nevents log ({len(events)} writes):")
    for ev in events:
        print(f"  {ev['seq']:2d}  {ev['sim_time']}  {ev['operation']}"
              f"  {ev['entity_type']}:{ev['entity_id']}  {ev['payload']}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            fh.write(result.model_dump_json(indent=2))
        print(f"\nwrote {args.json_out}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
