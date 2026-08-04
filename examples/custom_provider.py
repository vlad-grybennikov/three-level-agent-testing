"""Template: adding a model provider the codebase has never heard of.

The whole contract is one method, complete_json(system, user) -> dict,
plus one registration line. Stages, loop, harness, chat, and runner are
provider-blind. They only ever see the protocol.

Steps:
  1. copy this file, implement complete_json with your provider's SDK,
  2. import the module once at startup (registration is the import side
     effect),
  3. write a config file naming your provider, and pass it via --config.

Config file for this example:

    {
      "model": {
        "provider": "my_provider",
        "model": "some-model-id",
        "api_key": "env:MY_PROVIDER_API_KEY",
        "extra_options": {"region": "eu-west-1"}
      }
    }
"""

from telecom_aut.llm import LLMOutputError, _resolve_api_key, extract_json, register_provider


@register_provider("my_provider")
class MyProviderClient:
    def __init__(self, model_cfg) -> None:
        self.model = model_cfg.model
        self.api_key = _resolve_api_key(model_cfg.api_key)
        # Provider-specific knobs ride in extra_options (hashed with config):
        self.region = model_cfg.extra_options.get("region")
        # self.sdk = your_sdk.Client(api_key=self.api_key, region=self.region)

    def complete_json(self, system: str, user: str) -> dict:
        # reply_text = self.sdk.chat(model=self.model, system=system, user=user)
        # Reuse the shared tolerant parser so every provider is judged by the
        # same output-enforcement mechanism (comparability across models):
        # return extract_json(reply_text)
        raise NotImplementedError("wire your SDK call here")
