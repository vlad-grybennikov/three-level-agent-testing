// Single place the backend origin is named. Override with
// NEXT_PUBLIC_API_BASE when the Python server runs elsewhere.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8765";

export async function fetchState() {
  const res = await fetch(`${API_BASE}/state`);
  if (!res.ok) throw new Error(`GET /state -> ${res.status}`);
  return res.json();
}

export async function resetEnvironment() {
  const res = await fetch(`${API_BASE}/reset`, { method: "POST" });
  if (!res.ok) throw new Error(`POST /reset -> ${res.status}`);
  return res.json();
}

// POST /chat and invoke onEvent(parsedEvent) for every NDJSON line.
export async function streamChat(message, onEvent) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error(`POST /chat -> ${res.status}`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, i);
      buffer = buffer.slice(i + 1);
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
}
