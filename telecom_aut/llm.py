"""Single point of model access: the provider adapter registry.

Every pipeline stage speaks to the model through one interface:
`complete_json(system, user) -> dict`. Which backend serves it is purely
ModelConfig.provider. Model access is one swappable config value:

  "openai_compat": ANY OpenAI-style chat endpoint, whether local Ollama
                   (default), api.openai.com for GPT models, or any
                   compatible server. Differences are config values
                   (base_url, api_key, model).
  "anthropic":     the Claude API through the official anthropic SDK.

Adding a provider = one adapter (a callable taking ModelConfig, returning an
object with complete_json) + @register_provider("name"). Nothing else in the
codebase changes: stages, loop, harness, chat are all provider-blind.

api_key values of the form "env:VAR_NAME" are resolved from the environment
at client-build time, so config files never contain secrets and config_hash
covers the variable name, not the key.

Tests substitute a fake client implementing the same protocol, which is what
makes every stage unit-testable on frozen inputs with no model running.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Protocol, runtime_checkable


class LLMOutputError(Exception):
    """The model's reply contained no parseable JSON object."""


@runtime_checkable
class JSONChatClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply.

    Tolerates reasoning-model artifacts: <think> blocks, code fences, and
    prose around the object. Raises LLMOutputError if nothing parses.
    """
    cleaned = _THINK_RE.sub("", text).strip()

    fenced = _FENCE_RE.search(cleaned)
    candidates = [fenced.group(1)] if fenced else []

    # First balanced {...} span, string-aware.
    start = cleaned.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(cleaned[start:i + 1])
                    break

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise LLMOutputError(f"no JSON object in model reply: {text[:200]!r}")


def _content_text(content) -> str:
    """LangChain message content may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class ChatModelClient:
    """JSONChatClient over any LangChain chat model."""

    def __init__(self, chat_model) -> None:
        self.chat_model = chat_model

    def complete_json(self, system: str, user: str) -> dict:
        from langchain_core.messages import HumanMessage, SystemMessage

        reply = self.chat_model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return extract_json(_content_text(reply.content))


# -- provider registry ------------------------------------------------------

PROVIDERS: dict[str, Callable] = {}


def register_provider(name: str):
    """Register a model-provider adapter under a config-addressable name."""

    def wrap(factory):
        PROVIDERS[name] = factory
        return factory

    return wrap


def _resolve_api_key(value: str) -> str:
    """Support "env:VAR_NAME" indirection so keys stay out of config files."""
    if value.startswith("env:"):
        var = value[4:]
        resolved = os.environ.get(var)
        if not resolved:
            raise RuntimeError(
                f"api_key references environment variable {var!r}, "
                "which is not set"
            )
        return resolved
    return value


# Local reasoning_effort values -> Claude output_config.effort levels.
# "none" has no Claude equivalent (thinking is on by default on current
# models and disabling it invites tag-leak/plain-text-tool-call failure
# modes). The closest available substitute is adaptive thinking at low effort.
_CLAUDE_EFFORT = {"none": "low", "low": "low", "medium": "medium",
                  "high": "high"}


@register_provider("anthropic")
class AnthropicClient:
    """JSONChatClient over the official Anthropic SDK (the hosted path).

    Deliberately mechanism-identical to the local path: prompted JSON +
    tolerant extraction, no schema-constrained decoding, so comparing
    models never conflates provider with output-enforcement mechanism.

    Server-side refusal fallbacks (config: refusal_fallbacks, default on):
    a safety-classifier decline is re-run on Anthropic's recommended
    substitute model in the same call. Because a silent substitution would
    corrupt model attribution in the study, every fallback-served response
    is recorded in self.fallback_events and announced on stdout: rescued,
    but never invisible. A refusal that survives the whole chain raises
    LLMOutputError -> a recorded episode error.
    """

    def __init__(self, model_cfg) -> None:
        import anthropic

        # Zero-arg client: resolves ANTHROPIC_API_KEY or an `ant auth login`
        # profile from the environment. The config's api_key is Ollama-only.
        try:
            self.client = anthropic.Anthropic(
                timeout=model_cfg.timeout, max_retries=model_cfg.max_retries
            )
        except TypeError as exc:  # SDK: could not resolve authentication
            raise RuntimeError(
                "Claude credentials not found. Either export "
                "ANTHROPIC_API_KEY, put ANTHROPIC_API_KEY=... in the "
                "repo-root .env (the CLIs auto-load it), or run "
                "`ant auth login`."
            ) from exc
        self.model = model_cfg.model
        self.effort = _CLAUDE_EFFORT.get(
            model_cfg.reasoning_effort or "high", "high"
        )
        # max_tokens caps thinking + text together on current Claude models.
        # The local 1000-token guard would truncate mid-JSON, which reads as
        # a fake defect signal. Floor at 4096.
        self.max_tokens = max(model_cfg.max_tokens or 0, 4096)
        self.refusal_fallbacks = getattr(model_cfg, "refusal_fallbacks", True)
        self.fallback_events: list[dict] = []  # attribution stays visible

    def complete_json(self, system: str, user: str) -> dict:
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            output_config={"effort": self.effort},
            messages=[{"role": "user", "content": user}],
        )
        if self.refusal_fallbacks:
            response = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **kwargs,
            )
        else:
            response = self.client.messages.create(**kwargs)

        if response.stop_reason == "refusal":
            raise LLMOutputError(
                "model declined the request (stop_reason=refusal; the whole "
                "fallback chain refused)" if self.refusal_fallbacks else
                "model declined the request (stop_reason=refusal)"
            )
        # Covers sticky-routed turns too: a fallback_message iteration means
        # a fallback model served this response.
        iterations = getattr(response.usage, "iterations", None) or []
        if any(getattr(e, "type", None) == "fallback_message"
               for e in iterations):
            event = {"requested": self.model, "served_by": response.model}
            self.fallback_events.append(event)
            print(f"[claude] refusal fallback: served by {response.model} "
                  f"instead of {self.model}", flush=True)

        text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return extract_json(text)


@register_provider("openai_compat")
def _openai_compat_client(model_cfg) -> ChatModelClient:
    """Any OpenAI-style chat endpoint: Ollama, api.openai.com (GPT), etc."""
    from langchain_openai import ChatOpenAI

    kwargs = {}
    if getattr(model_cfg, "reasoning_effort", None) is not None:
        kwargs["reasoning_effort"] = model_cfg.reasoning_effort
    if getattr(model_cfg, "max_tokens", None) is not None:
        kwargs["max_tokens"] = model_cfg.max_tokens
    if getattr(model_cfg, "json_mode", False):
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
    chat = ChatOpenAI(
        model=model_cfg.model,
        base_url=model_cfg.base_url,
        api_key=_resolve_api_key(model_cfg.api_key),
        temperature=model_cfg.temperature,
        seed=model_cfg.seed,
        timeout=model_cfg.timeout,
        max_retries=model_cfg.max_retries,
        **kwargs,
    )
    return ChatModelClient(chat)


def build_chat_client(model_cfg):
    """ModelConfig -> live client, via the provider registry."""
    provider = getattr(model_cfg, "provider", "openai_compat")
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown model provider {provider!r}; available: "
            f"{sorted(PROVIDERS)}. Register custom adapters with "
            "telecom_aut.llm.register_provider."
        )
    return PROVIDERS[provider](model_cfg)
