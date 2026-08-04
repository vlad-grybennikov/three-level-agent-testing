"""Provider dispatch for the model client: local (openai_compat) vs Claude
(anthropic, official SDK). All offline: no network, no real credentials."""

import json
from pathlib import Path

import pytest

from telecom_aut.config import AgentConfig, ModelConfig
from telecom_aut.llm import AnthropicClient, build_chat_client, extract_json

CONFIGS = Path(__file__).parent.parent.parent / "configs"


def test_default_provider_is_local_openai_compat():
    cfg = ModelConfig()
    assert cfg.provider == "openai_compat"
    assert cfg.model == "qwen3.5"


def test_claude_config_file_loads_and_changes_hash():
    claude = AgentConfig.from_file(CONFIGS / "claude.json")
    assert claude.model.provider == "anthropic"
    assert claude.model.model == "claude-opus-5"
    assert claude.config_hash() != AgentConfig().config_hash()


def test_build_dispatches_to_anthropic_client(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    cfg = AgentConfig.from_file(CONFIGS / "claude.json").model
    client = build_chat_client(cfg)
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-opus-5"
    # reasoning_effort "low" -> Claude effort "low", floor on max_tokens.
    assert client.effort == "low"
    assert client.max_tokens == 4096


def test_effort_mapping_never_disables_thinking(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    # The local baseline "none" (no thinking) maps to adaptive-at-low-effort
    # on Claude, never thinking-disabled.
    cfg = ModelConfig(provider="anthropic", model="claude-opus-5",
                      reasoning_effort="none")
    assert AnthropicClient(cfg).effort == "low"
    cfg = ModelConfig(provider="anthropic", model="claude-opus-5",
                      reasoning_effort=None)
    assert AnthropicClient(cfg).effort == "high"


def test_local_max_tokens_guard_is_raised_for_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    # max_tokens caps thinking + text on Claude. The local 1000 guard would
    # truncate mid-JSON and read as a fake defect signal.
    cfg = ModelConfig(provider="anthropic", model="claude-opus-5",
                      max_tokens=1000)
    assert AnthropicClient(cfg).max_tokens == 4096
    cfg = ModelConfig(provider="anthropic", model="claude-opus-5",
                      max_tokens=8000)
    assert AnthropicClient(cfg).max_tokens == 8000


def test_extract_json_strips_thinking_tags_both_spellings():
    assert extract_json('<think>hmm</think>{"a": 1}') == {"a": 1}
    assert extract_json('<thinking>hmm</thinking>{"a": 1}') == {"a": 1}


def test_unknown_provider_fails_at_build_with_available_list():
    cfg = ModelConfig(provider="gemini")  # config accepts any name...
    with pytest.raises(ValueError, match="anthropic.*openai_compat"):
        build_chat_client(cfg)            # ...the registry validates at build


def test_refusal_fallbacks_flag(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    on = AnthropicClient(ModelConfig(provider="anthropic",
                                     model="claude-opus-5"))
    assert on.refusal_fallbacks is True          # user-chosen default
    assert on.fallback_events == []              # attribution ledger
    off = AnthropicClient(ModelConfig(provider="anthropic",
                                      model="claude-opus-5",
                                      refusal_fallbacks=False))
    assert off.refusal_fallbacks is False


def test_api_key_env_indirection(monkeypatch):
    from telecom_aut.llm import _resolve_api_key
    monkeypatch.setenv("SOME_PROVIDER_KEY", "sk-resolved")
    assert _resolve_api_key("env:SOME_PROVIDER_KEY") == "sk-resolved"
    assert _resolve_api_key("literal-key") == "literal-key"
    monkeypatch.delenv("MISSING_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        _resolve_api_key("env:MISSING_KEY")


def test_custom_adapter_runs_a_full_episode_with_no_other_changes():
    """The extensibility guarantee: register a brand-new provider, name it
    in config, and the agent runs an entire episode through it, with no
    edits to stages, loop, harness, or chat."""
    from telecom_aut.agent import TelecomAgent
    from telecom_aut.environment import TelecomEnv
    from telecom_aut.llm import PROVIDERS, register_provider
    from telecom_aut.testing.fakes import ScriptedLLM

    @register_provider("test_custom")
    class CustomAdapter(ScriptedLLM):
        def __init__(self, model_cfg):
            assert model_cfg.extra_options == {"region": "eu-test-1"}
            super().__init__([
                {"intent": "view_appointments"},
                {"subscriber_name": "Carol Okafor"},
                {"ranking": ["find_subscriber"]},
                {"name_or_email": "Carol Okafor"},
                {"ranking": ["finish"]},
                {"summary": "Carol's visit is APT-0401 on August 4."},
            ])

    try:
        config = AgentConfig.model_validate({"model": {
            "provider": "test_custom",
            "model": "totally-new-model",
            "extra_options": {"region": "eu-test-1"},
        }})
        env = TelecomEnv.fresh()
        agent = TelecomAgent(env, config)   # llm built FROM CONFIG via registry
        result = agent.run_episode("When is Carol Okafor's visit?")
        env.close()
        assert result.status == "finished"
        assert "APT-0401" in result.summary
        assert result.model_id == "totally-new-model"
    finally:
        PROVIDERS.pop("test_custom", None)  # keep the registry clean
