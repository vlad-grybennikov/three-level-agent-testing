"""Agent loop: a LangGraph state machine over the four pipeline stages.

Topology (single-turn episode):

    classify ──(unsupported)──> unsupported ─> END
        │
     extract ─> select ─> bind ──(bound)──> execute ──(finished)──> END
                  ^         │(bind error)      │
                  │         v                  │(budget left)
                  └──── budget check <─────────┘
                            │(budget spent)
                          exhaust ─> END

LangGraph is wiring only: every node body delegates to a stage object that is
independently constructable and testable. The loop owns three
things the stages must not: tool execution, the step budget, and episode
status.
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import AgentConfig
from .environment import TelecomEnv
from .llm import JSONChatClient, build_chat_client
from .pipeline import (
    FINISH_TOOL,
    BindingError,
    EntityExtractor,
    IntentClassifier,
    SelectionInput,
    SelectionResult,
    SlotSet,
    ToolSelector,
    load_binder_class,
)
from .retrieval import Retriever
from .tools import TelecomAPI, ToolRejection
from .tools import invoke as tool_invoke

_OBS_LIMIT = 600  # chars of tool result fed back into working history


TRACE_SCHEMA_VERSION = 1


class AgentState(TypedDict, total=False):
    instruction: str
    intent: str
    slots: dict
    history: list[str]          # compact per-step lines the LLM stages see
    steps: list[dict]           # structured records (feed the trace)
    snapshots: list[dict]       # {"index": step_index, "snapshot": {...}} per tool call
    iterations: int             # select-cycles consumed (the step budget)
    status: str                 # running | finished | unsupported | budget_exhausted
    summary: Optional[str]
    pending_selection: Optional[dict]
    pending_call: Optional[dict]


class EpisodeResult(BaseModel):
    instruction: str
    intent: str
    slots: dict
    steps: list[dict] = Field(default_factory=list)
    status: str
    summary: Optional[str] = None
    config_hash: str
    model_id: str
    # Versioned trace (schema in `trace["trace_schema_version"]`): initial
    # snapshot, per-step stage outputs + post-call snapshot, final snapshot,
    # config/model/seed, wall-clock timing. The harness consumes this.
    trace: dict = Field(default_factory=dict)


def _short(text: str, limit: int = _OBS_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 12] + f"...(+{len(text) - limit + 12} chars)"


class TelecomAgent:
    """Wires stages + tools + budget into a runnable episode."""

    def __init__(
        self,
        env: TelecomEnv,
        config: AgentConfig | None = None,
        llm: JSONChatClient | None = None,
    ) -> None:
        self.env = env
        self.config = config or AgentConfig()
        self.llm = llm if llm is not None else build_chat_client(self.config.model)
        self.api = TelecomAPI(env, retriever=Retriever(env, self.config.retrieval))
        self.classifier = IntentClassifier(self.llm, self.config)
        self.extractor = EntityExtractor(self.llm, self.config)
        self.selector = ToolSelector(self.llm, self.config)
        self.binder = load_binder_class(self.config.binder_class)(
            self.llm, self.config
        )
        self.graph = self._build_graph()

    # -- graph nodes --------------------------------------------------------

    def _sel_input(self, state: AgentState) -> SelectionInput:
        return SelectionInput(
            intent=state["intent"],
            slots=SlotSet.model_validate(state.get("slots", {})),
            history=state.get("history", []),
        )

    def _classify(self, state: AgentState) -> dict:
        return {"intent": self.classifier.run(state["instruction"]).intent}

    def _extract(self, state: AgentState) -> dict:
        slots = self.extractor.run(state["instruction"], state["intent"])
        return {"slots": slots.model_dump()}

    def _select(self, state: AgentState) -> dict:
        selection = self.selector.run(self._sel_input(state))
        return {
            "pending_selection": selection.model_dump(),
            "iterations": state["iterations"] + 1,
        }

    def _bind(self, state: AgentState) -> dict:
        selection = SelectionResult.model_validate(state["pending_selection"])
        tool = selection.top1
        try:
            call = self.binder.run(tool, self._sel_input(state))
            return {"pending_call": call.model_dump()}
        except BindingError as exc:
            record = {
                "index": len(state["steps"]),
                "selection": state["pending_selection"],
                "call": None,
                "result": None,
                "error": f"binding failed: {exc}",
            }
            return {
                "pending_call": None,
                "steps": state["steps"] + [record],
                "history": state["history"] + [f"bind {tool} FAILED: {exc}"],
            }

    def _execute(self, state: AgentState) -> dict:
        call = state["pending_call"]
        record: dict[str, Any] = {
            "index": len(state["steps"]),
            "selection": state["pending_selection"],
            "call": call,
            "result": None,
            "error": None,
        }
        if call["tool"] == FINISH_TOOL:
            record["result"] = {"finished": True}
            return {
                "steps": state["steps"] + [record],
                "status": "finished",
                "summary": call["args"].get("summary", ""),
                "pending_call": None,
            }
        try:
            result = tool_invoke(self.api, call["tool"], call["args"])
            record["result"] = result
            observation = _short(json.dumps(result, default=str))
        except ToolRejection as exc:
            record["error"] = str(exc)
            observation = f"REJECTED {exc}"
        line = f"{call['tool']}({json.dumps(call['args'])}) -> {observation}"
        # State is snapshotted after every tool call. The
        # trace interleaves state with operations (rejections included:
        # their snapshot proves nothing changed).
        snapshot = {"index": record["index"], "snapshot": self.env.snapshot()}
        return {
            "steps": state["steps"] + [record],
            "history": state["history"] + [line],
            "snapshots": state.get("snapshots", []) + [snapshot],
            "pending_call": None,
        }

    def _unsupported(self, state: AgentState) -> dict:
        return {
            "status": "unsupported",
            "summary": "Request is outside the supported capabilities "
                       "(subscriber info, view appointments, reschedule "
                       "appointment, service update, cancel order).",
        }

    def _exhaust(self, state: AgentState) -> dict:
        return {"status": "budget_exhausted"}

    # -- graph wiring -------------------------------------------------------

    def _build_graph(self):
        # One graph node per pipeline decision component, plus the non-LLM
        # execute_tool node. Node names mirror the stage names in pipeline/.
        g = StateGraph(AgentState)
        g.add_node("classify_intent", self._classify)
        g.add_node("extract_entities", self._extract)
        g.add_node("select_tool", self._select)
        g.add_node("bind_arguments", self._bind)
        g.add_node("execute_tool", self._execute)
        g.add_node("unsupported", self._unsupported)
        g.add_node("exhaust", self._exhaust)

        def budget_route(state: AgentState) -> str:
            if state.get("status") == "finished":
                return "done"
            if state["iterations"] >= self.config.max_steps:
                return "exhaust"
            return "select_tool"

        g.add_edge(START, "classify_intent")
        g.add_conditional_edges(
            "classify_intent",
            lambda s: "unsupported" if s["intent"] == "unsupported"
            else "extract_entities",
            {"unsupported": "unsupported", "extract_entities": "extract_entities"},
        )
        g.add_edge("extract_entities", "select_tool")
        g.add_edge("select_tool", "bind_arguments")
        g.add_conditional_edges(
            "bind_arguments",
            lambda s: "execute_tool" if s.get("pending_call") else budget_route(s),
            {"execute_tool": "execute_tool", "select_tool": "select_tool",
             "exhaust": "exhaust", "done": END},
        )
        g.add_conditional_edges(
            "execute_tool", budget_route,
            {"select_tool": "select_tool", "exhaust": "exhaust", "done": END},
        )
        g.add_edge("unsupported", END)
        g.add_edge("exhaust", END)
        return g.compile()

    # -- entry point --------------------------------------------------------

    def run_episode(self, instruction: str, on_state=None) -> EpisodeResult:
        """Run one episode. `on_state`, if given, receives the full agent
        state after every graph node. Callers use it for live progress."""
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        t0 = time.monotonic()
        initial_snapshot = self.env.snapshot()
        initial: AgentState = {
            "instruction": instruction,
            "history": [],
            "steps": [],
            "snapshots": [],
            "iterations": 0,
            "status": "running",
            "summary": None,
        }
        final: dict = initial
        for state in self.graph.stream(
            initial,
            config={"recursion_limit": 4 * self.config.max_steps + 12},
            stream_mode="values",
        ):
            final = state
            if on_state is not None:
                on_state(state)

        steps = final.get("steps", [])
        snap_by_index = {
            s["index"]: s["snapshot"] for s in final.get("snapshots", [])
        }
        # finish / bind-failure steps executed no tool: snapshot is None.
        trace_steps = [
            dict(step, post_call_snapshot=snap_by_index.get(step["index"]))
            for step in steps
        ]
        trace = {
            "trace_schema_version": TRACE_SCHEMA_VERSION,
            "instruction": instruction,
            "model_id": self.config.model.model,
            "config_hash": self.config.config_hash(),
            "seed": self.config.model.seed,
            "timing": {
                "started_at": started_at,
                "duration_s": round(time.monotonic() - t0, 3),
            },
            "intent": final.get("intent", "unsupported"),
            "slots": final.get("slots", {}),
            "initial_snapshot": initial_snapshot,
            "steps": trace_steps,
            "final_snapshot": self.env.snapshot(),
            "status": final.get("status", "running"),
            "summary": final.get("summary"),
        }
        return EpisodeResult(
            instruction=instruction,
            intent=final.get("intent", "unsupported"),
            slots=final.get("slots", {}),
            steps=steps,
            status=final.get("status", "running"),
            summary=final.get("summary"),
            config_hash=self.config.config_hash(),
            model_id=self.config.model.model,
            trace=trace,
        )
