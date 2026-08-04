"""HTTP-level tests for chat.py, the primary tested entry point.

A real ChatServer runs on an ephemeral port with a scripted LLM. Tests speak
actual HTTP: NDJSON streaming, transcript folding, reset, per-interaction
log files, CORS, and concurrency guarding. No model server involved.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from telecom_aut.agent import TelecomAgent
from telecom_aut.chat import ChatServer
from telecom_aut.config import AgentConfig
from telecom_aut.environment import TelecomEnv

from telecom_aut.testing.fakes import ScriptedLLM

# Two consecutive episodes: an info question, then a follow-up that relies
# on the folded conversation context for the subscriber's identity.
SCRIPT = [
    # episode 1: "What is Vlad Grybennikov's current order?"
    {"intent": "subscriber_info"},
    {"subscriber_name": "Vlad Grybennikov"},
    {"ranking": ["find_subscriber"]},
    {"name_or_email": "Vlad Grybennikov"},
    {"ranking": ["list_orders"]},
    {"subscriber_id": "SUB-0007"},
    {"ranking": ["finish"]},
    {"summary": "Vlad's active order is ORD-0108 (Fiber 1 Gig)."},
    # episode 2: "what time is his visit?"  (identity from folded context)
    {"intent": "view_appointments"},
    {"subscriber_name": "Vlad Grybennikov"},
    {"ranking": ["list_appointments"]},
    {"subscriber_id": "SUB-0007", "status": "pending"},
    {"ranking": ["finish"]},
    {"summary": "His visit APT-0407 is on August 4 at 11:00."},
]


@pytest.fixture()
def server(tmp_path):
    env = TelecomEnv.fresh()
    llm = ScriptedLLM(list(SCRIPT))
    agent = TelecomAgent(env, AgentConfig(), llm=llm)
    srv = ChatServer(("127.0.0.1", 0), env, agent, tmp_path)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv, base, llm, tmp_path
    srv.shutdown()
    srv.server_close()
    env.close()


def get(base, path):
    with urllib.request.urlopen(base + path) as res:
        return res.status, dict(res.headers), json.loads(res.read())


def post_chat(base, message):
    req = urllib.request.Request(
        base + "/chat",
        data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as res:
        body = res.read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_full_chat_flow_over_http(server):
    srv, base, llm, log_dir = server

    # -- state before anything happens
    status, headers, state = get(base, "/state")
    assert status == 200
    assert headers.get("Access-Control-Allow-Origin") == "*"
    assert len(state["subscribers"]) == 7
    assert all(s["full_name"] and "@" in s["email"]
               for s in state["subscribers"])  # demo roster needs both
    assert state["events"] == []

    # -- episode 1: plain question, streamed NDJSON
    events = post_chat(base, "What is Vlad Grybennikov's current order?")
    types = [e["type"] for e in events]
    assert types[0] == "intent" and types[-1] == "final"
    assert events[-1]["status"] == "finished"
    assert "ORD-0108" in events[-1]["summary"]

    # -- episode 2: follow-up, identity must come from folded context
    events2 = post_chat(base, "what time is his visit?")
    assert events2[-1]["status"] == "finished"
    assert "APT-0407" in events2[-1]["summary"]
    # The intent-classification prompt of episode 2 carried the context:
    ep2_intent_prompt = llm.calls[8][1]
    assert "Earlier conversation" in ep2_intent_prompt
    assert "Current request: what time is his visit?" in ep2_intent_prompt
    assert "ORD-0108" in ep2_intent_prompt  # ep1 reply is part of context

    # -- both interactions were logged as JSON files
    logs = sorted(log_dir.glob("*-ep*.json"))
    assert len(logs) == 2
    first = json.loads(logs[0].read_text())
    assert first["user_message"].startswith("What is Vlad")
    assert first["instruction"] == first["user_message"]  # no context yet
    assert first["result"]["status"] == "finished"
    second = json.loads(logs[1].read_text())
    assert second["user_message"] == "what time is his visit?"
    assert "Earlier conversation" in second["instruction"]

    # -- reads only: no state mutation from either episode
    _, _, state_after = get(base, "/state")
    assert state_after["events"] == []

    # -- reset clears environment AND transcript
    req = urllib.request.Request(base + "/reset", data=b"{}", method="POST")
    with urllib.request.urlopen(req) as res:
        assert json.loads(res.read()) == {"ok": True}
    assert srv.transcript == []


def test_empty_message_is_400(server):
    _, base, _, _ = server
    req = urllib.request.Request(
        base + "/chat", data=json.dumps({"message": "  "}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400


def test_concurrent_episode_gets_409(server):
    srv, base, _, _ = server
    assert srv.episode_lock.acquire(blocking=False)  # simulate a running episode
    try:
        req = urllib.request.Request(
            base + "/chat", data=json.dumps({"message": "hi"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req)
        assert exc.value.code == 409
    finally:
        srv.episode_lock.release()


def test_cors_preflight(server):
    _, base, _, _ = server
    req = urllib.request.Request(base + "/chat", method="OPTIONS")
    with urllib.request.urlopen(req) as res:
        assert res.status == 204
        assert res.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in res.headers.get("Access-Control-Allow-Methods", "")
