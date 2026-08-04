# Testing AI Agents at Three Levels

Companion code for the paper *Testing AI Agents at Three Levels: An
Evaluation Harness and Case Study*. It applies the classical testing
hierarchy to an LLM agent:

- Level 1, end-to-end: judges the run's outcome. The projected final
  database state must match the annotated goal state. Each task runs k
  times and aggregates as pass^k.
- Level 2, integration: judges the run's path. Executed tool calls are
  scored against a dependency graph (required operations, value-matched
  edges, ordering) plus hard policy assertions.
- Level 3, component: judges the pipeline stages in isolation on frozen
  inputs. Metrics: intent F1, slot match, selection Recall@k, and
  argument binding exactness.

An optional LLM judge grades the same episodes from their trajectories
into a separate `judge_pass_hat_k` column. The deterministic oracle runs
either way, so judge-vs-oracle agreement is a measured number, not an
assumption.

## The agent under test

A telecom care-ops assistant built as a four-stage pipeline (intent
classification, entity extraction, tool selection, argument binding),
wired as a LangGraph loop over permissive CRUD pseudo-APIs and a
deterministic SQLite environment. Policy rules live in a retrieval
corpus, not in code. The agent can violate them, and the harness
measures which level detects each failure.

## Task annotations

Each task JSON in `approach_tests/tasks/` carries the annotations for
all three levels at once:

```
approach_tests/tasks/<task>.json
├── instruction + variants    the input: what the agent is asked
├── goal + projection         Level 1: expected final DB state
├── dependency + policies     Level 2: required ops, value-matched
│                             edges, ordering, hard assertions
└── level3                    Level 3: frozen per-stage fixtures
```

The runner plays k live episodes per task. Every trace is graded by
Level 1 and Level 2 at the same time, which is what makes disagreement
between the levels observable. Level 3 never runs an episode. It calls
each pipeline stage once, in isolation, on the frozen inputs. The
labeled intent set in `approach_tests/fixtures/level3_intents.json`
(72 utterances, 12 per class) supplements Level 3 with enough examples
for per-class intent F1 (`--intent-dataset`).

## Layout

```
telecom_aut/          the agent and the harness (environment, tools,
                      pipeline stages, LangGraph loop, testing/ evaluators)
approach_tests/       the paper's tests: 9 annotated tasks, the intent
                      set, scripted episodes, one test file per level
tests/unit/           ordinary unit tests for the codebase itself
configs/              model configs, the only thing that changes per model
report-*.json         suite reports with full traces
report_summary_*.ipynb   analysis notebooks, one per report
export_assets.py      turns report JSON into tables and figures
chat-ui/              Next.js demo UI for the chat server
```

[STRUCTURE.md](STRUCTURE.md) has the full folder-by-folder map, the
task annotation anatomy, and the scripted episode catalogue.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
make test          # full offline suite, no model needed
make help          # all task shortcuts
```

Live episodes need a model. Local baseline through Ollama:

```bash
ollama pull qwen3.5
make suite MODEL=qwen K=3
```

Hosted Claude, with `ANTHROPIC_API_KEY=...` in `.env` or the
environment:

```bash
make suite MODEL=opus K=3 FLAGS="--judge --variants \
    --intent-dataset approach_tests/fixtures/level3_intents.json"
```

Each run writes `report-<model>.json` with per-episode traces, verdicts
from all three levels, judge verdicts, and pass^k. Analysis is offline
from there:

```bash
make notebook NB=report_summary_opus.ipynb   # or report_summary_qwen.ipynb
make export    # tables and figures into exports/
```

Chat demo: `make chat` in one terminal, `make ui` in another, then open
http://localhost:3011.

## Swapping models

Providers are small adapters registered in `telecom_aut/llm.py`. The
whole contract is `complete_json(system, user) -> dict` plus one
`@register_provider` line. Two ship with the repo: any OpenAI-compatible
endpoint (Ollama, api.openai.com, vLLM) and the Claude API. See
[configs/README.md](configs/README.md) and
`examples/custom_provider.py`. Secrets never live in config files.
`"api_key": "env:VAR_NAME"` resolves from the environment.

## Injecting defects

All five defect surfaces are config-only, no source edits: system
prompt, tool descriptions, selector config, retrieval config, and the
pluggable argument binder. Every run records a config hash, so each
trace is attributable to an exact configuration.

## Growing the task set

`make variants` asks an LLM to paraphrase task instructions, and `make
intents` does the same for the labeled intent set. An entity guard
rejects any paraphrase that loses a name, date, or id. Everything
generated lands with `"reviewed": false` and stays out of every
evaluation until a human flips it. `make review-list` shows what is
waiting. This is what keeps the intent set citable as human-verified.

## License

MIT. See [LICENSE](LICENSE).
