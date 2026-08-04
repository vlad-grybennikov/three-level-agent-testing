# Model configs

Which model serves an episode is decided entirely by the `model` block
of the config file passed with `--config`. Tasks, evaluators, and traces
stay the same, so cross-model comparisons hold the harness constant.

| File | Provider | Model |
|---|---|---|
| `local-qwen.json` (also the no-config default) | `openai_compat` | `qwen3.5` via local Ollama, temperature 0 plus a fixed seed |
| `claude.json` | `anthropic` | `claude-opus-5` through the official SDK. Auth from `ANTHROPIC_API_KEY`. Refusal fallbacks on, recorded and announced when they fire |
| `gpt.json` | `openai_compat` | GPT via api.openai.com, same adapter as Ollama. Auth via `env:OPENAI_API_KEY`. Set `model` to the tag your account serves |

Notes:

- Secrets never go in these files. `api_key: "env:VAR_NAME"` resolves
  from the environment when the client is built. The config hash covers
  the variable name, not the key.
- `reasoning_effort` means different things per provider. On Ollama it
  maps to the endpoint's reasoning toggle ("none" disables thinking).
  On Claude it maps to `output_config.effort` with adaptive thinking,
  never disabled. On OpenAI reasoning models, use the values the API
  accepts.
- Temperature and seed apply to `openai_compat` only. The Claude API
  rejects sampling parameters, and pass^k absorbs the stochasticity.
- Adding a provider takes one adapter with
  `complete_json(system, user) -> dict`, one
  `@telecom_aut.llm.register_provider("name")` line, and a config file
  naming it. No other code changes.
