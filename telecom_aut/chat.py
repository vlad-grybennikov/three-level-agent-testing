"""Chat API server over the agent: a demo/dev tool, NOT a study surface.

    python -m telecom_aut.chat [--port 8765] [--db path] [--config cfg.json]
                               [--model tag] [--reasoning none|low|medium|high]
                               [--log-dir chat_logs]

The web UI lives in chat-ui/ (a Next.js app): `cd chat-ui && npm run dev`,
then open http://localhost:3011. This process serves only JSON/NDJSON (with
permissive CORS for the UI) and:
  * prints exact per-step logs (intent, slots, candidate rankings, calls,
    results) to its terminal,
  * writes one JSON file per interaction under --log-dir, containing the
    full event stream plus the EpisodeResult.

Design notes:
  * The agent under test is single-turn. Each chat message runs
    one independent episode. Continuity between messages comes only from
    the persistent database state.
  * This module adds no behaviour to the AUT. One episode at a time (409).

Endpoints:
  GET  /:      service info (JSON)
  POST /chat:  {"message": str} -> NDJSON stream of progress events
  POST /reset: reset environment to seed
  GET  /state: compact snapshot (subscribers, orders, appointments, events)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import TelecomAgent
from .config import AgentConfig
from .environment import TelecomEnv


def compose_instruction(transcript: list[tuple[str, str]], message: str,
                        max_exchanges: int = 6) -> str:
    """Fold recent chat exchanges into a single-turn instruction.

    The AUT stays single-turn: it receives ONE instruction string per
    episode. Conversation continuity is a chat-layer convenience. The
    composed string is what gets logged, so demo traces remain auditable.
    """
    if not transcript:
        return message
    lines = ["Earlier conversation (context only; records may have changed"
             " since):"]
    for user_msg, agent_reply in transcript[-max_exchanges:]:
        lines.append(f"Operator: {user_msg}")
        lines.append(f"Agent: {agent_reply}")
    lines.append("")
    lines.append(f"Current request: {message}")
    return "\n".join(lines)


def run_streaming_episode(agent: TelecomAgent, message: str, emit):
    """Run one episode, calling emit(dict) for each progress event.

    Returns the EpisodeResult (or None if the episode raised). Factored out
    of the HTTP layer so it is testable with a scripted LLM. Event sequence:
    intent -> slots -> step* -> final (or error).
    """
    sent = {"intent": False, "slots": False, "steps": 0}

    def on_state(state: dict) -> None:
        if not sent["intent"] and state.get("intent"):
            sent["intent"] = True
            emit({"type": "intent", "intent": state["intent"]})
        if not sent["slots"] and state.get("slots") is not None:
            sent["slots"] = True
            filled = {k: v for k, v in state["slots"].items() if v is not None}
            emit({"type": "slots", "slots": filled})
        steps = state.get("steps") or []
        while sent["steps"] < len(steps):
            step = steps[sent["steps"]]
            sent["steps"] += 1
            candidates = [
                c["tool"]
                for c in (step.get("selection") or {}).get("candidates", [])
            ]
            result = step.get("result")
            result_json = None
            if result is not None and (step.get("call") or {}).get("tool") != "finish":
                result_json = json.dumps(result, default=str)
                if len(result_json) > 300:
                    result_json = result_json[:300] + "..."
            emit({
                "type": "step",
                "index": step["index"],
                "candidates": candidates,
                "call": step.get("call"),
                "result": result_json,
                "error": step.get("error"),
            })

    try:
        result = agent.run_episode(message, on_state=on_state)
        emit({
            "type": "final",
            "status": result.status,
            "summary": result.summary,
            "intent": result.intent,
            "steps": len(result.steps),
        })
        return result
    except Exception as exc:  # endpoint down, model error, ...
        emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        return None


def print_event(event: dict, prefix: str = "") -> None:
    """Terminal renderer for one streamed event (shared chat/step logging)."""
    kind = event.get("type")
    if kind == "intent":
        print(f"{prefix}intent: {event['intent']}", flush=True)
    elif kind == "slots":
        print(f"{prefix}slots:  {json.dumps(event['slots'])}", flush=True)
    elif kind == "step":
        ranked = " > ".join(event.get("candidates") or [])
        print(f"{prefix}step {event['index']:2d}  candidates: {ranked}",
              flush=True)
        call = event.get("call")
        if call:
            print(f"{prefix}         call: {call['tool']}"
                  f"({json.dumps(call['args'])})", flush=True)
        if event.get("error"):
            print(f"{prefix}         error: {event['error']}", flush=True)
        elif event.get("result"):
            out = event["result"]
            print(f"{prefix}         result: {out[:200]}"
                  f"{'...' if len(out) > 200 else ''}", flush=True)
    elif kind == "final":
        print(f"{prefix}status: {event['status']}"
              + (f": {event['summary']}" if event.get("summary") else ""),
              flush=True)
    elif kind == "error":
        print(f"{prefix}ERROR: {event['message']}", flush=True)


class ChatHandler(BaseHTTPRequestHandler):
    # HTTP/1.0: close-delimited bodies make chunk-free streaming trivial.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # keep the terminal for step logs only
        pass

    @property
    def ctx(self):
        return self.server  # ChatServer with env/agent/lock/log_dir

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # CORS preflight for the Next.js UI
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    # -- routes -------------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/":
            self._json(200, {
                "service": "telecom-aut chat API",
                "ui": "cd chat-ui && npm run dev  ->  http://localhost:3011",
                "endpoints": ["POST /chat", "POST /reset", "GET /state"],
            })
        elif self.path == "/state":
            snap = self.ctx.env.snapshot()["tables"]
            self._json(200, {
                "subscribers": [
                    {"id": s["id"], "full_name": s["full_name"],
                     "email": s["email"], "state": s["state"]}
                    for s in snap["subscribers"]
                ],
                "plans": [{"id": p["id"], "code": p["code"]}
                          for p in snap["plans"]],
                "orders": snap["orders"],
                "appointments": snap["appointments"],
                "slots": {s["id"]: s["starts_at"]
                          for s in snap["availability_slots"]},
                "events": snap["events"],
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/reset":
            with self.ctx.episode_lock:
                self.ctx.env.reset_to_seed()
                self.ctx.transcript.clear()
            print("[chat] environment and transcript reset", flush=True)
            self._json(200, {"ok": True})
            return
        if self.path != "/chat":
            self._json(404, {"error": "not found"})
            return

        message = (self._read_body().get("message") or "").strip()
        if not message:
            self._json(400, {"error": "empty message"})
            return
        if not self.ctx.episode_lock.acquire(blocking=False):
            self._json(409, {"error": "an episode is already running"})
            return
        try:
            n = self.ctx.next_episode_number()
            instruction = compose_instruction(self.ctx.transcript, message)
            print(f"\n[chat] episode {n}: {message}", flush=True)
            if instruction != message:
                print(f"  (context: {len(self.ctx.transcript)} earlier"
                      " exchanges folded into the instruction)", flush=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            dead = False
            events: list[dict] = []

            def emit(obj: dict) -> None:
                nonlocal dead
                events.append(obj)
                print_event(obj, prefix="  ")       # exact steps -> terminal
                if dead:
                    return
                try:
                    self.wfile.write((json.dumps(obj) + "\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    dead = True  # client left, let the episode finish cleanly

            result = run_streaming_episode(self.ctx.agent, instruction, emit)
            reply = (result.summary or f"[{result.status}]") if result \
                else "[error]"
            self.ctx.transcript.append((message, reply))
            self.ctx.write_log(n, message, instruction, events, result)
        finally:
            self.ctx.episode_lock.release()


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, env: TelecomEnv, agent: TelecomAgent,
                 log_dir: Path) -> None:
        super().__init__(addr, ChatHandler)
        self.env = env
        self.agent = agent
        self.episode_lock = threading.Lock()
        self.log_dir = log_dir
        self._episode_counter = 0
        # (user_message, agent_reply) pairs, folded into later instructions.
        self.transcript: list[tuple[str, str]] = []

    def next_episode_number(self) -> int:
        self._episode_counter += 1
        return self._episode_counter

    def write_log(self, n: int, message: str, instruction: str,
                  events: list[dict], result) -> None:
        """One JSON file per interaction: the streamed events + full result."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self.log_dir / f"{stamp}-ep{n:03d}.json"
        payload = {
            "user_message": message,
            "instruction": instruction,  # composed: context + current request
            "logged_at": stamp,
            "model_id": self.agent.config.model.model,
            "config_hash": self.agent.config.config_hash(),
            "events": events,
            "result": result.model_dump() if result is not None else None,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  log written: {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    from .envfile import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(prog="telecom_aut.chat")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--reasoning", default=None,
                        choices=["none", "low", "medium", "high"])
    parser.add_argument("--log-dir", default="chat_logs",
                        help="directory for per-interaction JSON logs")
    args = parser.parse_args(argv)

    config = AgentConfig.from_file(args.config) if args.config else AgentConfig()
    if args.model or args.reasoning:
        config = config.model_copy(deep=True)
        if args.model:
            config.model.model = args.model
        if args.reasoning:
            config.model.reasoning_effort = args.reasoning

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = TelecomEnv.fresh(args.db)
    agent = TelecomAgent(env, config)
    server = ChatServer(("127.0.0.1", args.port), env, agent, log_dir)
    print(f"chat API: http://127.0.0.1:{args.port}/ (UI: cd chat-ui && npm run dev)")
    print(f"model:    {config.model.model}  (config {config.config_hash()})")
    print(f"logs:     {log_dir}/  (one JSON per interaction; steps echo here)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
