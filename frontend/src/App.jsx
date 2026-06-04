import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./App.css";
import logo from "./logo.png";

const API_URL = "http://localhost:8000";

const SUGGESTIONS = [
  "What is paracetamol used for?",
  "Side effects of Lisinopril?",
  "Can I take metformin with alcohol?",
  "What is amoxicillin prescribed for?",
  "What is ibuprofen used for?",
  "Drug interactions with warfarin?",
];

function Dots() {
  return (
    <span className="dots">
      <span />
      <span />
      <span />
    </span>
  );
}

function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div
      className={`bubble-row ${isUser ? "bubble-row--user" : "bubble-row--ai"}`}
    >
      {!isUser && <img src={logo} className="ai-avatar" alt="Pharmacy AI" />}
      <div className="bubble-col">
        <div className={`bubble ${isUser ? "bubble--user" : "bubble--ai"}`}>
          {msg.loading ? <Dots /> : msg.content}
        </div>
        {msg.meta && !msg.loading && (
          <div className="meta-row">
            {msg.meta.mode === "rag" && (
              <>
                <span
                  className={`meta-badge ${msg.meta.has_interaction ? "meta-badge--warn" : "meta-badge--ok"}`}
                >
                  {msg.meta.has_interaction ? "⚠ Interaction" : "✓ Safe"}
                </span>
                <span className="meta-badge meta-badge--dim">
                  {msg.meta.chunks_used} sources
                </span>
              </>
            )}
            {msg.meta.mode === "agent" && (
              <>
                <span
                  className={`meta-badge ${msg.meta.has_interaction ? "meta-badge--warn" : "meta-badge--ok"}`}
                >
                  {msg.meta.has_interaction ? "⚠ Interaction" : "✓ Safe"}
                </span>
                <span className="meta-badge meta-badge--dim">
                  {msg.meta.sources_used} sources
                </span>
                <span className="meta-badge meta-badge--confidence">
                  {msg.meta.confidence === "High"
                    ? "●"
                    : msg.meta.confidence === "Medium"
                      ? "◑"
                      : "○"}{" "}
                  {msg.meta.confidence}
                </span>
              </>
            )}
            <span className="meta-badge meta-badge--dim">
              {msg.meta.elapsed_ms}ms
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [mode, setMode] = useState("rag");

  const bottomRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function resize() {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
  }

  async function send(q) {
    const question = (q || input).trim();
    if (!question || loading) return;
    setInput("");
    if (taRef.current) taRef.current.style.height = "40px";

    setMessages((prev) => [
      ...prev,
      { role: "user", content: question },
      { role: "ai", loading: true, content: "" },
    ]);
    setLoading(true);

    const currentMode = mode;

    try {
      if (currentMode === "rag") {
        const { data } = await axios.post(`${API_URL}/ask`, {
          question,
          top_k: 5,
        });
        setMessages((prev) => {
          const u = [...prev];
          u[u.length - 1] = {
            role: "ai",
            content: data.answer,
            meta: {
              mode: "rag",
              chunks_used: data.chunks_used,
              has_interaction: data.has_interaction,
              elapsed_ms: data.elapsed_ms,
            },
          };
          return u;
        });
      } else {
        const { data } = await axios.post(`${API_URL}/agent/ask`, { question });
        const hasInteraction =
          data.answer.toLowerCase().includes("major") ||
          data.answer.toLowerCase().includes("interaction");
        setMessages((prev) => {
          const u = [...prev];
          u[u.length - 1] = {
            role: "ai",
            content: data.answer,
            meta: {
              mode: "agent",
              sources_used: data.sources_used,
              confidence: data.confidence,
              has_interaction: hasInteraction,
              elapsed_ms: data.elapsed_ms,
            },
          };
          return u;
        });
      }

      setHistory((prev) => [
        { q: question, t: new Date().toLocaleTimeString() },
        ...prev.slice(0, 9),
      ]);
    } catch (err) {
      const msg = err.response?.data?.detail || "Connection error.";
      setMessages((prev) => {
        const u = [...prev];
        u[u.length - 1] = { role: "ai", content: `❌ ${msg}` };
        return u;
      });
    } finally {
      setLoading(false);
      taRef.current?.focus();
    }
  }

  function onKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="side">
        <div className="side-logo">
          <img src={logo} className="side-logo-img" alt="logo" />
          <span className="side-logo-name">Pharmacy Assistant</span>
        </div>
        <div className="side-block">
          <div className="side-label">Recent queries</div>
          {history.length === 0 ? (
            <div className="side-empty">No queries yet</div>
          ) : (
            history.map((h, i) => (
              <button key={i} className="side-item" onClick={() => send(h.q)}>
                <span className="side-item-q">{h.q}</span>
                <span className="side-item-t">{h.t}</span>
              </button>
            ))
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="main">
        {/* Topbar — toggle + clear */}
        <div className="topbar">
          <div className="mode-toggle">
            <button
              className={`mode-btn ${mode === "rag" ? "mode-btn--active" : ""}`}
              onClick={() => setMode("rag")}
              title="Direct RAG retrieval — fast, precise"
            >
              RAG
            </button>
            <button
              className={`mode-btn ${mode === "agent" ? "mode-btn--active" : ""}`}
              onClick={() => setMode("agent")}
              title="ReAct Agent — multi-step reasoning with tools"
            >
              Agent
            </button>
          </div>
          <button className="clear-btn" onClick={() => setMessages([])}>
            Clear chat
          </button>
        </div>

        {/* Chat */}
        <div className="chat">
          {messages.length === 0 && (
            <div className="welcome">
              <img src={logo} className="welcome-logo" alt="Pharmacy AI" />
              <h1 className="welcome-h">How can I help you?</h1>
              <p className="welcome-p">
                Ask about medications, interactions, side effects, or dosage.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s, i) => (
                  <button
                    key={i}
                    className="suggestion"
                    onClick={() => send(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <Bubble key={i} msg={m} />
          ))}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="input-zone">
          <div className="input-box">
            <textarea
              ref={taRef}
              className="input-ta"
              placeholder={
                mode === "rag"
                  ? "Ask about a medication or drug interaction…"
                  : "Ask a complex question — agent will reason step by step…"
              }
              value={input}
              onChange={(e) => {
                setInput(e.target.value);
                resize();
              }}
              onKeyDown={onKey}
              rows={1}
              disabled={loading}
            />
            <button
              className={`send ${loading ? "send--busy" : ""}`}
              onClick={() => send()}
              disabled={loading || !input.trim()}
              aria-label="Send"
            >
              ↑
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
