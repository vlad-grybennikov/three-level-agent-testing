# Three-level agent tests

The paper's agent tests, kept separate from the ordinary unit tests in
`tests/unit/`. The unit tests answer "is the code correct?". This
directory answers "is the agent good, and which level of testing can
tell?"

Each task JSON in `tasks/` annotates all three levels at once. One live
trace is graded by Levels 1 and 2 together. Level 3 calls the pipeline
stages in isolation and never runs an episode.

Contents:

- `test_level1_e2e.py`: outcome checks, projected final DB state vs the
  goal state, pass^k.
- `test_level2_integration.py`: path checks, dependency graph and hard
  policy assertions.
- `test_level3_components.py`: per-stage checks on frozen fixtures.
- `test_judge_example.py`: LLM-as-judge wiring and fail-closed parsing.
- `test_variants_example.py`, `test_intent_augment.py`: paraphrase
  generation and the human review gate.
- `tasks/`: 9 annotated tasks covering all five capabilities in both
  polarities, mutations and refusals where the right outcome is no
  change.
- `fixtures/level3_intents.json`: 72 reviewed labeled utterances, 12
  per class. Unreviewed generations are excluded by the loader.
- `episodes.py`: scripted episodes that drive the real agent loop with
  a scripted LLM, including engineered defects and the
  layer-disagreement specimens.

Everything here runs offline and deterministically (`make test-levels`).
The same task files drive live runs:

    make suite MODEL=qwen K=3 FLAGS="--judge --variants \
        --intent-dataset approach_tests/fixtures/level3_intents.json"

See [STRUCTURE.md](../STRUCTURE.md) for the per-task table, the task
JSON anatomy, and the specimen catalogue.
