# Three-Level Agent Tests (paper §3)

This directory holds the paper's agent tests, separated from the ordinary
Python unit tests in `tests/` (which cover the codebase itself:
environment, tools, pipeline plumbing, chat API). The rule of thumb:
`tests/unit/` answers *"is the code correct?"*; this directory answers
*"is the agent good — and which level of testing can tell?"*

Each task JSON in `tasks/` annotates all three levels at once:
`goal`+`projection` feed Level 1, `dependency`+`policies` feed Level 2,
and the `level3` block holds the frozen per-stage component fixtures. One
live trace is graded by Levels 1 and 2 simultaneously; Level 3 calls the
stages in isolation and never runs an episode (see the root README for
the full mapping).

The level test files:

| File | Level | Judges | Against |
|---|---|---|---|
| `test_level1_e2e.py` | 1 — end-to-end | the run's **outcome** | projected final DB state vs G; pass^k |
| `test_level2_integration.py` | 2 — integration | the run's **path** | D (ops, value-matched edges, ordering) + P (hard assertions) |
| `test_level3_components.py` | 3 — component | the agent's **components** | frozen inputs: intent F1, slot match, Recall@k, binding exactness |

Plus the stochastic-side examples:

| File | What it demonstrates |
|---|---|
| `test_judge_example.py` | Classic LLM-as-judge wiring: trajectory evidence, fail-closed parsing (incl. truncated judge replies — a live-run regression), and a `judge_pass_hat_k` column independent of the Level-1 oracle. |
| `test_variants_example.py` | Instruction-paraphrase pipeline: entity guard, human review gate, runner rotation over reviewed phrasings sharing one `(G, D, P)`. |
| `test_intent_augment.py` | The same generate → human-review gate for the labeled intent dataset. |

Supporting data:

- `tasks/` — **9 annotated tasks**, one `(G, D, P, level3, variants)` JSON
  each, covering all five capabilities *in both polarities*: mutating happy
  paths (cancel with cleanup, plan upgrade, reschedule slot-swap), policy
  refusals where the correct goal state is *no change* (`G` empty +
  `forbid_call`) — one whose forbidding fact must be actively looked up
  (unpaid invoices) and one visible on the read path (disallowed plan) —
  a nothing-to-do negative control, and two read-only queries. See
  [STRUCTURE.md](../STRUCTURE.md) for the per-task table and JSON anatomy.
- `fixtures/level3_intents.json` — **72 reviewed labeled utterances**
  (12 per class): 18 author seeds + 54 human-reviewed LLM paraphrases.
  Unreviewed generations are excluded from every evaluation by the loader —
  the author-only set saturates (F1 = 1.0 across models); the full set
  separates them.
- `episodes.py` — scripted episodes that drive the *real* agent loop and
  environment (only the LLM is scripted): compliant runs of every task
  shape, engineered defects (`BAD_CANCEL`, `WRONG_ORDER_CANCEL`,
  `WRONG_TARGET_CANCEL_DEV`, `BAD_REFUSAL_BRUNO`), and the
  layer-disagreement specimens in both directions — outcome-right /
  path-wrong (`INVENTED_SLOT_RESCHEDULE`) and path-clean / outcome-wrong
  (`WRONG_REGION_RESCHEDULE`).

Everything here is offline and deterministic (`make test-levels`). The same
task files drive live runs against a real model:

    make suite MODEL=qwen K=3 FLAGS="--judge --variants \
        --intent-dataset approach_tests/fixtures/level3_intents.json"
