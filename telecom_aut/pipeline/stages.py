"""The four pipeline stages.

Every stage:
  * is constructed from (llm_client, AgentConfig) with no environment access,
  * exposes run(frozen_input) -> typed_output,
  * touches no database and no other stage.

Stage 4 (argument binding) is pluggable: the loop instantiates whatever class
`AgentConfig.binder_class` names (defect surface #5). Any binder must expose
run(tool, SelectionInput) -> BoundCall.
"""

from __future__ import annotations

import importlib
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..prompts import BIND_PROMPT, EXTRACT_PROMPT, INTENT_PROMPT, SELECT_PROMPT
from ..tools import TOOL_NAMES, TOOL_REGISTRY
from .types import (
    INTENTS,
    BoundCall,
    IntentResult,
    SelectionInput,
    SelectionResult,
    SlotSet,
    ToolCandidate,
)

FINISH_TOOL = "finish"


class FinishArgs(BaseModel):
    """Arguments of the loop-level finish pseudo-tool."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1)


def _render_history(history: list[str], limit: int = 12) -> str:
    """Render working history, collapsing consecutive identical lines.

    The collapse matters: at temperature 0 a run of verbatim-repeated
    call->result lines is a pattern the model will keep continuing
    (observed empirically as list_appointments loops). One line with an
    explicit repeat count breaks the pattern and names the pathology.
    """
    if not history:
        return "(none yet)"
    collapsed: list[tuple[str, int]] = []
    for line in history:
        if collapsed and collapsed[-1][0] == line:
            collapsed[-1] = (line, collapsed[-1][1] + 1)
        else:
            collapsed.append((line, 1))
    rendered = [
        line if count == 1
        else f"{line}  [repeated {count}x with identical result — do not repeat]"
        for line, count in collapsed
    ]
    return "\n".join(rendered[-limit:])


class IntentClassifier:
    """Stage 1: utterance -> intent."""

    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.system = config.system_prompt

    def run(self, utterance: str) -> IntentResult:
        data = self.llm.complete_json(
            self.system, INTENT_PROMPT.format(utterance=utterance)
        )
        intent = data.get("intent") if isinstance(data, dict) else None
        if not isinstance(intent, str) or intent.strip() not in INTENTS:
            # Anything unparseable is, by definition, not one of the three
            # supported capabilities.
            return IntentResult(intent="unsupported")
        return IntentResult(intent=intent.strip())


class EntityExtractor:
    """Stage 2: (utterance, intent) -> SlotSet."""

    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.system = config.system_prompt

    def run(self, utterance: str, intent: str) -> SlotSet:
        data = self.llm.complete_json(
            self.system, EXTRACT_PROMPT.format(utterance=utterance, intent=intent)
        )
        if not isinstance(data, dict):
            return SlotSet()
        # Drop individually malformed slots rather than failing the stage:
        # a wrong-typed slot becomes a *missing* slot, which is a measurable
        # extraction miss, not a crash.
        for _ in range(2):
            try:
                return SlotSet.model_validate(data)
            except ValidationError as exc:
                bad = {e["loc"][0] for e in exc.errors() if e["loc"]}
                if not bad:
                    return SlotSet()
                data = {k: v for k, v in data.items() if k not in bad}
        return SlotSet()


class ToolSelector:
    """Stage 3: SelectionInput -> ranked top-k tool candidates.

    Knobs (defect surface #3, SelectorConfig): top_k, excluded_tools
    (candidate filtering), invert_ranking (scoring perturbation).
    """

    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.system = config.system_prompt
        self.cfg = config.selector
        self.descriptions = config.effective_tool_descriptions()
        self.space = [
            t for t in TOOL_NAMES + [FINISH_TOOL]
            if t not in self.cfg.excluded_tools
        ]

    def run(self, sel_input: SelectionInput) -> SelectionResult:
        catalog = "\n".join(
            f"- {t}: {self.descriptions.get(t, '')}" for t in self.space
        )
        data = self.llm.complete_json(
            self.system,
            SELECT_PROMPT.format(
                intent=sel_input.intent,
                slots=sel_input.slots.model_dump_json(),
                catalog=catalog,
                history=_render_history(sel_input.history),
                top_k=self.cfg.top_k,
            ),
        )
        raw = data.get("ranking") if isinstance(data, dict) else None
        seen: set[str] = set()
        ranking = []
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, str) and item in self.space and item not in seen:
                seen.add(item)
                ranking.append(item)
        if not ranking:
            ranking = [FINISH_TOOL]  # graceful stall-out, never a crash
        ranking = ranking[: self.cfg.top_k]
        if self.cfg.invert_ranking:
            ranking.reverse()
        n = len(ranking)
        return SelectionResult(
            candidates=[
                ToolCandidate(tool=t, score=round((n - i) / n, 4))
                for i, t in enumerate(ranking)
            ]
        )


class BindingError(Exception):
    """Stage 4 could not produce a schema-valid argument object."""


class LLMArgumentBinder:
    """Stage 4 (clean baseline): (tool, SelectionInput) -> BoundCall.

    Output is validated against the tool's Pydantic schema. Only keys the
    model actually set are forwarded (partial-update semantics survive).
    """

    def __init__(self, llm, config) -> None:
        self.llm = llm
        self.system = config.system_prompt
        self.descriptions = config.effective_tool_descriptions()

    def _args_model(self, tool: str) -> type[BaseModel]:
        if tool == FINISH_TOOL:
            return FinishArgs
        if tool not in TOOL_REGISTRY:
            raise BindingError(f"unknown tool {tool!r}")
        return TOOL_REGISTRY[tool][0]

    def run(self, tool: str, sel_input: SelectionInput) -> BoundCall:
        model = self._args_model(tool)
        data = self.llm.complete_json(
            self.system,
            BIND_PROMPT.format(
                tool=tool,
                description=self.descriptions.get(tool, ""),
                schema=json.dumps(model.model_json_schema(), indent=None),
                intent=sel_input.intent,
                slots=sel_input.slots.model_dump_json(),
                history=_render_history(sel_input.history),
            ),
        )
        try:
            validated = model.model_validate(data)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
                for e in exc.errors()
            )
            raise BindingError(f"{tool} arguments invalid: {details}") from exc
        return BoundCall(
            tool=tool, args=validated.model_dump(exclude_unset=True)
        )


def load_binder_class(path: str) -> type:
    """Resolve 'pkg.module:ClassName' (or dotted), defect surface #5."""
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(f"no attribute {attr!r} in {module_name}") from exc
