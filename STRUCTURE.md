# Repository structure

`telecom_aut/` holds the agent under test plus the harness that measures
it. `approach_tests/` holds the annotated tasks and the three-level test
suite from the paper. The root holds configuration, run evidence, and
analysis tooling.

```
telecom_aut/          agent + harness code
approach_tests/       the paper's tests: tasks, fixtures, level tests
tests/unit/           ordinary unit tests for the codebase itself
configs/              per-model run configs
examples/             how to plug in a new model provider
chat-ui/              Next.js demo UI (not a study surface)
report-opus.json      committed suite reports, full traces included
report-qwen.json
report_summary_opus.ipynb   analysis notebook for the Claude report
report_summary_qwen.ipynb   analysis notebook for the qwen report
export_assets.py      report JSON -> CSV/MD tables + PNG/PDF figures
final-matrices-generation.py   per-task matrix figures for the paper
media/                committed figures used in the paper
exports/              generated tables and figures (gitignored, `make export`)
Makefile              task shortcuts (`make help`)
```

## telecom_aut/: the agent and the harness

| Path | What it is |
|---|---|
| `environment/` | Deterministic SQLite world. `schema.sql` and `seed.py` build 7 subscribers, 9 orders, 8 appointments, and 190 availability slots, byte-identical across processes. `env.py` adds a simulated clock (+1 s per write, rejected calls tick nothing). |
| `tools/` | 15 permissive CRUD pseudo-APIs (`api.py`). Rejection is structural only (not_found, ambiguous, invalid_state, invalid_args). Policy violations execute. `descriptions.py` holds the tool descriptions the selector reads, one of the defect surfaces. |
| `pipeline/` | The four LLM stages (`stages.py`): `classify_intent`, `extract_entities`, `select_tool`, `bind_arguments`. Each is independently callable, which is what Level 3 relies on. |
| `agent.py` | LangGraph loop wiring the stages over the tools. Captures the versioned episode trace: snapshots before and after every call, ranked selection candidates, timing. |
| `prompts.py` | `DEFAULT_SYSTEM_PROMPT` (a defect surface, operating procedure only, since eligibility rules live in the policy corpus and must be retrieved) plus the fixed per-stage templates. |
| `retrieval.py` | BM25 over the 12-document policy corpus (8 load-bearing, 4 distractors). `k`, `k1`, `b`, and `category_filter` come from config and are also a defect surface. |
| `config.py` | `AgentConfig`: model plus the five defect surfaces, hashed into `config_hash` so every trace points at an exact configuration. |
| `llm.py` | Provider adapter registry. Contract: `complete_json(system, user) -> dict` plus `@register_provider`. Ships `openai_compat` (Ollama, OpenAI, vLLM) and `anthropic` (official SDK, optional refusal fallbacks recorded as `fallback_events`). `"api_key": "env:VAR"` keeps secrets out of configs. |
| `envfile.py` | Minimal `.env` loader for the CLI entry points. Never overrides shell vars, never logs values. |
| `testing/` | The harness, see the next table. |
| `chat.py`, `run.py` | Demo chat API server and single-episode CLI. Demo layer, not a study surface. |

### telecom_aut/testing/: the three-level harness

| File | Role |
|---|---|
| `annotations.py` | The task schema. `TaskAnnotation` holds G (goal state as cell-level patches over the projected initial state, everything else must stay unchanged), D (dependency graph: required ops, value-matched edges, ordering), P (hard policy assertions: `require_before`, `forbid_call`), the Level 3 fixtures, instruction variants, and the `Projection` Level 1 applies before comparing states. Loaders enforce the review gate, so unreviewed LLM generations never reach an evaluation. |
| `levels.py` | The evaluators. Level 1: projected-state diff plus τ-bench pass^k. Level 2: node and edge precision/recall/F1 (earliest-producer edge attribution), order conformance, redundancy ratio, policy assertions. Level 3: per-component metrics on frozen inputs. |
| `judge.py` | LLM-as-judge. Grades episode success from the trajectory (request, executed actions, projected state delta, final reply) into `judge_pass_hat_k`, a column parallel to Level 1 that never replaces it. Fails closed on unparseable replies. |
| `runner.py` | Episodes to traces to per-level verdicts to one report JSON with full traces, so all analysis re-derives offline. Flags: `--k`, `--config`, `--judge`, `--judge-model`, `--variants`, `--intent-dataset`, `--out`. |
| `variants.py` | LLM instruction paraphrasing with a deterministic entity guard (names, dates, and ids must survive). Writes everything with `reviewed: false`. |
| `intent_augment.py` | The same pattern for the labeled intent dataset. Paraphrases inherit the seed case's label and humans flip `reviewed`. |
| `fakes.py` | `ScriptedLLM` and `LoopingLLM`, the offline stand-ins that make the framework testable without a model. |

## approach_tests/: the paper's test suite

Unit tests in `tests/unit/` check the codebase. This directory checks
agent behavior and demonstrates the framework on engineered episodes
with known ground truth.

### tasks/: nine annotated tasks

Each JSON file is one task: an operator instruction plus the full
(G, D, P, level3) annotation and reviewed phrasing variants.

```jsonc
{
  "id": "cancel-vlad-order",
  "capability": "cancel_order",
  "instruction": "Cancel Vlad Grybennikov's internet order.",
  "variants": [ {"text": "...", "source": "llm", "reviewed": true} ],
  "goal":      { /* G: cell patches, empty means "change nothing" */ },
  "dependency":{ /* D: required_ops (+args_include), edges, ordering */ },
  "policies":  [ /* P: require_before / forbid_call, hard pass or fail */ ],
  "level3":    { /* frozen fixtures: expected_intent, expected_slots,
                    selection_cases, binding_cases */ },
  "annotation":{ "annotator": "...", "minutes": 12 }
}
```

The set covers all five capabilities in both polarities: tasks where the
right answer is a change and tasks where it is no change.

| Task | Capability | What it probes |
|---|---|---|
| `cancel-vlad.json` | cancel_order | Full cancellation chain: invoice check, pending-visit cleanup, and slot release before the order cancel. |
| `cancel-dev-pending.json` | cancel_order | Cancellation on a multi-order subscriber: binding disambiguation plus cleanup. |
| `cancel-bruno-refusal.json` | cancel_order | Refusal: an unpaid balance forbids cancellation. G is empty and `forbid_call(cancel_order)` applies. The forbidding fact sits in the invoices table, so the agent must choose to look. |
| `downgrade-alice-legacy.json` | service_update | Refusal: the requested plan is not offered. Also catches helpful substitution of a different plan. |
| `upgrade-erin.json` | service_update | Happy-path plan change, subscriber identified by email. |
| `reschedule-vlad.json` | reschedule_appointment | Slot swap: release the old slot, book the new one, update the appointment. |
| `reschedule-erin-negative.json` | reschedule_appointment | Negative control: nothing is pending, so the correct behavior is to change nothing and say so. |
| `info-dev-active-order.json` | subscriber_info | Read-only field query, G empty. |
| `view-carol-visit.json` | view_appointments | Read-only appointment view, G empty. |

### fixtures/level3_intents.json: the labeled intent set

72 reviewed utterances, 12 per intent class including `unsupported`.
18 are author-written seeds and 54 are LLM paraphrases that passed the
human review gate. Paraphrase labels are inherited from seed cases and
count as unverified until a human flips `"reviewed": true`. The loader
excludes unreviewed cases from every scored evaluation, so the benchmark
cannot silently become an echo of the generator. The harder set also
matters in practice: the 18 author seeds saturate at F1 = 1.0 for every
model tried, while the full 72 separate models.

### episodes.py: scripted episodes with known ground truth

Scripted tool-call sequences that drive the real agent loop,
environment, and trace capture. Only the LLM is replaced with
`ScriptedLLM`. Because each defect is engineered, each specimen proves
which level detects what.

| Specimen | Engineered behavior | Who catches it |
|---|---|---|
| `GOOD_*` (7 specimens) | Compliant runs of every task shape | all levels pass (baseline) |
| `BAD_CANCEL` | Lazy cancel: order cancelled, pending visit stranded | Level 1 (state diff) |
| `WRONG_ORDER_CANCEL` | Asks for confirmation only after the destructive write | Level 1 passes, Level 2 flags the ordering violation and the failed `require_before` |
| `WRONG_TARGET_CANCEL_DEV` | Multi-order subscriber, wrong target | Level 3 binding, which schema validity alone cannot catch |
| `INVENTED_SLOT_RESCHEDULE` | Books a slot id that was never retrieved | Level 1 passes, Level 2 edge recall and precision dip |
| `WRONG_REGION_RESCHEDULE` | Books a valid but wrong-region slot | Level 2 nearly clean, Level 1 fails |
| `REJECTION_RECOVERY_RESCHEDULE` | A rejected call, then recovery | nothing: rejected calls pollute no metric |
| `BAD_REFUSAL_BRUNO` | Cancels despite the unpaid-balance policy | Level 1 (G empty) plus the `forbid_call` assertion |

The disagreement specimens cover both directions: defects Level 2 flags
while Level 1 passes (`WRONG_ORDER_CANCEL`, `INVENTED_SLOT_RESCHEDULE`)
and one Level 1 flags while Level 2 is nearly clean
(`WRONG_REGION_RESCHEDULE`).

### The test files

| File | What it pins |
|---|---|
| `test_level1_e2e.py` | Goal-state evaluation: happy paths, refusals as empty G, the everything-else-unchanged rule, projection, pass^k arithmetic. |
| `test_level2_integration.py` | Path metrics on the specimens above: node and edge F1, earliest-producer attribution, ordering, redundancy, policy assertions, both disagreement directions. |
| `test_level3_components.py` | Each stage in isolation on frozen fixtures, through the configured binder, so an injected defective binder is measured rather than bypassed. |
| `test_judge_example.py` | Judge wiring: full trajectory evidence in the prompt, fail-closed parsing (including truncated judge replies, a live-run regression), and a `judge_pass_hat_k` column that cannot move the oracle. |
| `test_variants_example.py` | The entity guard, the review gate, and runner rotation over reviewed phrasings that share one (G, D, P). |
| `test_intent_augment.py` | Paraphrase generation guards and the dataset review gate. |

Everything in this directory runs offline and deterministically
(`make test-levels`). The same task files drive live runs against a real
model (`make suite MODEL=...`).

## tests/unit/: codebase unit tests

Ordinary tests for the codebase itself: environment determinism, tool
rejection semantics, retrieval determinism and injected-defect behavior,
pipeline stage contracts, agent-loop plumbing, provider adapters, chat
API, `.env` loading. A test that answers "is the code correct?" lives
here. "Is the agent good?" lives in `approach_tests/`.

## configs/ and examples/

`configs/*.json` are complete run configurations, so swapping models is
config-only. See [configs/README.md](configs/README.md).
`examples/custom_provider.py` shows the recipe for a new provider: one
adapter class plus one `@register_provider` line.

## Run evidence and analysis (repo root)

- `report-<model>.json`: one suite run each, with per-episode traces,
  all three levels, judge verdicts, and pass^k tables. Everything
  downstream re-derives from these files with zero model calls.
- `report_summary_opus.ipynb`, `report_summary_qwen.ipynb`: analysis
  notebooks, self-contained (pandas and matplotlib only).
- `export_assets.py` (`make export`): regenerates all tables (CSV and
  MD) and figures (PNG and PDF) into `exports/`, which is gitignored.
- `final-matrices-generation.py` (`make matrices`): rebuilds the
  per-task matrix figures in `media/` from the exported CSVs.
