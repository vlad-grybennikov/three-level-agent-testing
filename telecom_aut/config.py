"""Agent configuration: the defect-injection layer. Each of the five
defect surfaces has its own config slot:

  1. system_prompt:       swappable text (baseline: prompts.DEFAULT_SYSTEM_PROMPT)
  2. tool_descriptions:   per-tool overrides (baseline: DEFAULT_TOOL_DESCRIPTIONS)
  3. selector:            SelectorConfig (top_k, excluded_tools, invert_ranking)
  4. retrieval:           RetrievalConfig (k, k1, b, category_filter)
  5. binder_class:        dotted/colon path to the stage-4 implementation

`AgentConfig()` with no arguments IS the documented clean baseline.
Every run records config_hash() so a trace is attributable to an exact config.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .prompts import DEFAULT_SYSTEM_PROMPT
from .retrieval import RetrievalConfig
from .tools import DEFAULT_TOOL_DESCRIPTIONS


class ModelConfig(BaseModel):
    """The single swappable model-access value-set.

    provider "openai_compat" (default) targets the local Ollama endpoint via
    base_url/api_key. provider "anthropic" targets the Claude API through the
    official SDK: auth comes from the environment (ANTHROPIC_API_KEY or an
    `ant auth login` profile), and base_url / api_key / temperature / seed /
    json_mode are ignored (current Claude models reject sampling params).
    """

    model_config = ConfigDict(extra="forbid")

    # Validated against the adapter registry at client-build time, so custom
    # adapters registered via telecom_aut.llm.register_provider just work.
    provider: str = "openai_compat"
    base_url: str = "http://localhost:11434/v1"  # Ollama OpenAI-compatible
    api_key: str = "ollama"  # "env:VAR_NAME" resolves from the environment
    model: str = "qwen3.5"
    temperature: float = 0.0
    seed: int = 7
    timeout: float = 300.0
    max_retries: int = 1
    # Qwen3.5 thinks by default (~30x latency for stage-sized JSON tasks,
    # measured 27s -> 0.9s per call). "none" is the baseline. Set to null /
    # "low" / "high" to study the reasoning variant, a single config value.
    reasoning_effort: Optional[str] = "none"
    max_tokens: Optional[int] = 1000  # guard against runaway generation
    # Structured-output enforcement: when True, requests are sent with
    # response_format json_object so the backend constrains decoding to JSON.
    # Off by default until verified against the local endpoint. The tolerant
    # parser in llm.py is the fallback mechanism either way.
    json_mode: bool = False
    # anthropic provider only: server-side refusal fallbacks. Fallback-served
    # responses are recorded and announced, never silent (attribution).
    refusal_fallbacks: bool = True
    # Escape hatch for custom adapters: provider-specific options (region,
    # project id, native-API knobs, ...) that ModelConfig doesn't model.
    # Participates in config_hash like every other field.
    extra_options: dict = Field(default_factory=dict)


class SelectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(default=5, ge=1)
    excluded_tools: list[str] = Field(default_factory=list)
    invert_ranking: bool = False

    @field_validator("excluded_tools")
    @classmethod
    def _known_tools(cls, v: list[str]) -> list[str]:
        unknown = set(v) - set(DEFAULT_TOOL_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"unknown tools in excluded_tools: {sorted(unknown)}")
        return v


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    selector: SelectorConfig = Field(default_factory=SelectorConfig)
    binder_class: str = "telecom_aut.pipeline.stages:LLMArgumentBinder"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    tool_descriptions: dict[str, str] = Field(default_factory=dict)
    max_steps: int = Field(default=12, ge=1)

    @field_validator("tool_descriptions")
    @classmethod
    def _known_tools(cls, v: dict[str, str]) -> dict[str, str]:
        unknown = set(v) - set(DEFAULT_TOOL_DESCRIPTIONS)
        if unknown:
            raise ValueError(f"unknown tools in tool_descriptions: {sorted(unknown)}")
        return v

    def effective_tool_descriptions(self) -> dict[str, str]:
        merged = dict(DEFAULT_TOOL_DESCRIPTIONS)
        merged.update(self.tool_descriptions)
        return merged

    def config_hash(self) -> str:
        canonical = json.dumps(
            self.model_dump(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentConfig":
        return cls.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
