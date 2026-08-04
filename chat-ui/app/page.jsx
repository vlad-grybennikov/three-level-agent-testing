"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import StatePanel from "@/components/StatePanel";
import { fetchState, resetEnvironment, streamChat } from "@/lib/api";

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [state, setState] = useState(null);
  const logRef = useRef(null);

  const refreshState = useCallback(async () => {
    try {
      setState(await fetchState());
    } catch {
      setState(null);
    }
  }, []);

  useEffect(() => {
    refreshState();
  }, [refreshState]);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages]);

  // Update the last (agent) message in place while the episode streams.
  const patchLast = (patch) =>
    setMessages((all) => {
      const next = all.slice();
      next[next.length - 1] = { ...next[next.length - 1], ...patch };
      return next;
    });

  async function send(e) {
    e.preventDefault();
    const message = input.trim();
    if (!message || busy) return;
    setInput("");
    setMessages((all) => [
      ...all,
      { role: "user", text: message },
      { role: "agent", text: "…", error: false },
    ]);
    setBusy(true);
    let steps = 0;
    try {
      await streamChat(message, (ev) => {
        if (ev.type === "step") {
          steps += 1;
          patchLast({ text: `… (${steps} steps)` });
        } else if (ev.type === "final") {
          patchLast({
            text: ev.summary || `[${ev.status}] no reply text`,
            error: ev.status !== "finished",
          });
        } else if (ev.type === "error") {
          patchLast({ text: `error: ${ev.message}`, error: true });
        }
      });
    } catch (err) {
      patchLast({ text: `error: ${err.message}`, error: true });
    }
    setBusy(false);
    refreshState();
  }

  async function reset() {
    try {
      await resetEnvironment();
      setMessages((all) => [
        ...all,
        { role: "note", text: "— environment reset to seed —" },
      ]);
      refreshState();
    } catch (err) {
      setMessages((all) => [
        ...all,
        { role: "note", text: `reset failed: ${err.message}` },
      ]);
    }
  }

  return (
    <div className="shell">
      <main>
        <header>
          <h1>Telecom Chatbot Sample</h1>
          <button onClick={reset}>reset environment</button>
        </header>
        <div className="log" ref={logRef}>
          {messages.map((m, i) =>
            m.role === "note" ? (
              <div key={i} className="note">
                {m.text}
              </div>
            ) : (
              <div
                key={i}
                className={`msg ${m.role === "user" ? "user" : ""} ${
                  m.error ? "err" : ""
                }`}
              >
                {m.text}
              </div>
            )
          )}
        </div>
        <form onSubmit={send}>
          <input
            type="text"
            autoComplete="off"
            placeholder="e.g. Reschedule Vlad Grybennikov's visit to the afternoon of August 5."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button className="primary" disabled={busy} type="submit">
            Send
          </button>
        </form>
      </main>
      <StatePanel
        state={state}
        onPick={(text) =>
          setInput((v) => (v && !v.endsWith(" ") ? `${v} ${text}` : v + text))
        }
      />
    </div>
  );
}
