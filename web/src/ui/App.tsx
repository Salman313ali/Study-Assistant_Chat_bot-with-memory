import React, { useMemo, useRef, useState } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

async function chatApi(message: string, sessionId: string, style: "short" | "detailed") {
  const resp = await fetch(import.meta.env.VITE_API_URL ?? "http://localhost:8000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId, style }),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || `HTTP ${resp.status}`);
  }
  return (await resp.json()) as {
    answer: string;
    key_points: string[];
    suggested_questions: string[];
    references: string[];
  };
}

async function resetApi(sessionId: string) {
  const url = new URL(import.meta.env.VITE_API_URL ?? "http://localhost:8000/chat");
  // Replace /chat with /session/{id}
  const base = url.origin;
  const resetUrl = `${base}/session/${encodeURIComponent(sessionId)}`;
  const resp = await fetch(resetUrl, { method: "DELETE" });
  if (!resp.ok) throw new Error(await resp.text());
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [style, setStyle] = useState<"short" | "detailed">("short");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const listRef = useRef<HTMLDivElement>(null);

  async function send(override?: string) {
    const text = (override ?? input).trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", content: text }]);
    try {
      console.log("Sending with style:", style); // Debug log
      const resp = await chatApi(text, sessionId, style);
      const formatted = [
        resp.answer,
        resp.key_points.length ? `\n\nKey Points:\n- ${resp.key_points.join("\n- ")}` : "",
        // Intentionally omit Suggested Questions text to avoid duplication with buttons below
      ]
        .filter(Boolean)
        .join("");
      setMessages((m) => [...m, { role: "assistant", content: formatted }]);
      setSuggestions(resp.suggested_questions ?? []);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `Error: ${e?.message ?? "Something went wrong"}` },
      ]);
      setSuggestions([]);
    } finally {
      setBusy(false);
      listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
    }
  }

  return (
    <div className="min-h-dvh grid grid-rows-[auto_1fr_auto]">
      <header className="border-b bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/40">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h1 className="text-xl font-semibold">Study Assistant</h1>
              <p className="text-sm text-zinc-600">LangChain + Groq + HF + Chroma</p>
            </div>
            <div className="flex items-center gap-2">
              <div className="hidden sm:flex items-center gap-1 mr-2">
                <label className="text-sm text-zinc-600">Style:</label>
                <select
                  className="text-sm border rounded-md px-2 py-1 bg-white"
                  value={style}
                  onChange={(e) => setStyle(e.target.value as any)}
                  disabled={busy}
                >
                  <option value="short">Short</option>
                  <option value="detailed">Detailed</option>
                </select>
              </div>
              <button
                className="rounded-lg border px-3 py-1.5 text-sm bg-white hover:bg-zinc-50"
                onClick={async () => {
                  if (busy) return;
                  const nextId = crypto.randomUUID();
                  try {
                    await resetApi(sessionId);
                  } catch {
                    // ignore reset errors on client
                  }
                  setMessages([]);
                  setSuggestions([]);
                  setSessionId(nextId);
                }}
              >
                New Chat
              </button>
            </div>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-3xl w-full px-4 py-4">
        <div ref={listRef} className="space-y-4 overflow-y-auto max-h-[70dvh] pr-1">
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
              <div
                className={
                  "inline-block rounded-2xl px-4 py-3 whitespace-pre-wrap " +
                  (m.role === "user"
                    ? "bg-zinc-900 text-white"
                    : "bg-white border shadow-sm text-zinc-900")
                }
              >
                {m.content}
              </div>
            </div>
          ))}
          {suggestions.length ? (
            <div className="flex flex-wrap gap-2 mt-2">
              {suggestions.map((q, idx) => (
                <button
                  key={idx}
                  className="rounded-full bg-zinc-900 text-white px-3 py-1.5 text-sm shadow hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-zinc-400"
                  onClick={() => send(q)}
                  disabled={busy}
                >
                  {q}
                </button>
              ))}
            </div>
          ) : null}
          {!messages.length && (
            <div className="text-center text-zinc-500 py-16">
              Ask a question to get started.
            </div>
          )}
        </div>
      </main>
      <footer className="bg-white/70 backdrop-blur supports-[backdrop-filter]:bg-white/40 border-t">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <div className="flex items-center gap-2">
            <input
              className="flex-1 rounded-xl border px-3 py-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-zinc-400"
              placeholder="Type your message..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              disabled={busy}
            />
            <button
              className="rounded-xl bg-zinc-900 text-white px-4 py-2 disabled:opacity-60"
              onClick={() => send()}
              disabled={busy}
            >
              {busy ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}

