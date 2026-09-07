import { useCallback, useEffect, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api, onAgentEvent } from "./api";
import type {
  ApprovalCard,
  DiagRow,
  ModelInfo,
  ProviderRow,
  SessionRow,
  TaskSummary,
} from "./types";

interface UiMsg {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
}

interface ActivityItem {
  id: number;
  kind: "tool" | "output" | "ok" | "err";
  text: string;
}

interface KindMeta {
  label: string;
  blurb: string;
  base: string | null;
  baseHint: string;
  hints: string[];
}

const KINDS: Record<string, KindMeta> = {
  openai: {
    label: "OpenAI",
    blurb: "ChatGPT models. Needs an OpenAI API key.",
    base: null,
    baseHint: "https://api.openai.com/v1",
    hints: ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-cyber"],
  },
  anthropic: {
    label: "Anthropic",
    blurb: "Claude models. Needs an Anthropic API key.",
    base: null,
    baseHint: "(provider default)",
    hints: ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"],
  },
  xai: {
    label: "SpaceXAI",
    blurb: "Grok models by SpaceXAI. Needs an xAI API key.",
    base: null,
    baseHint: "(provider default)",
    hints: ["grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1"],
  },
  mistral: {
    label: "Mistral",
    blurb: "Mistral + Codestral. Needs a La Plateforme API key.",
    base: "https://api.mistral.ai/v1",
    baseHint: "https://api.mistral.ai/v1",
    hints: [
      "mistral-large-latest",
      "mistral-medium-latest",
      "mistral-small-latest",
      "codestral-latest",
      "devstral-latest",
    ],
  },
  gemini: {
    label: "Gemini",
    blurb: "Google Gemini. Needs a Google AI Studio key.",
    base: "https://generativelanguage.googleapis.com/v1beta/openai",
    baseHint: "https://generativelanguage.googleapis.com/v1beta/openai",
    hints: ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.1-pro-preview", "gemini-3.6-flash"],
  },
  ollama: {
    label: "Ollama",
    blurb: "Free models on your own PC via Ollama. No key — install Ollama and pull a model first.",
    base: "http://localhost:11434/v1",
    baseHint: "http://localhost:11434/v1",
    hints: ["llama4:scout", "qwen3:14b", "qwen3:8b", "llama3.1:8b", "mistral:7b", "llama3.2:3b"],
  },
  local: {
    label: "Local",
    blurb: "LM Studio, vLLM or llama.cpp server. No key — enter its address.",
    base: "http://localhost:1234/v1",
    baseHint: "http://localhost:1234/v1 (LM Studio) or :8000 (vLLM)",
    hints: [],
  },
};

const KIND_IDS = Object.keys(KINDS);
const EFFORTS = ["off", "low", "medium", "high", "max"];
const EFFORT_LABELS: Record<string, string> = {
  off: "Off",
  low: "Low",
  medium: "Medium",
  high: "High",
  max: "Max",
};

function catalogLine(m: ModelInfo): string {
  const parts: string[] = [];
  if (m.context_window != null) {
    parts.push(
      m.context_window >= 1_000_000
        ? `${(m.context_window / 1_000_000).toFixed(2)}M ctx`
        : `${Math.round(m.context_window / 1000)}K ctx`,
    );
  }
  if (m.input_price_per_mtok != null && m.output_price_per_mtok != null) {
    parts.push(`$${m.input_price_per_mtok}/$${m.output_price_per_mtok} per MTok`);
  }
  if (m.knowledge_cutoff) parts.push(`cutoff ${m.knowledge_cutoff}`);
  if (m.notes) parts.push(m.notes);
  return parts.length > 0 ? parts.join(" · ") : "no metadata";
}

export default function App() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMsg[]>([]);
  const [stream, setStream] = useState("");
  const [streamReasoning, setStreamReasoning] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [activity, setActivity] = useState<ActivityItem[]>([]);
  const [approvals, setApprovals] = useState<ApprovalCard[]>([]);
  const [providers, setProviders] = useState<ProviderRow[]>([]);
  const [keys, setKeys] = useState<Record<string, boolean>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [lastSummary, setLastSummary] = useState<TaskSummary | null>(null);
  const [mode, setModeState] = useState<"build" | "plan">("build");
  const [autonomy, setAutonomySel] = useState("low");
  const [effortSel, setEffortSel] = useState<string | null>(null);

  const [showProviders, setShowProviders] = useState(false);
  const [showModels, setShowModels] = useState(false);
  const [showDiags, setShowDiags] = useState(false);
  const [newWorkspace, setNewWorkspace] = useState("");
  const [draft, setDraft] = useState("");
  const [providerSel, setProviderSel] = useState("");
  const [modelSel, setModelSel] = useState("");
  const [micOn, setMicOn] = useState(false);

  const [browserProvider, setBrowserProvider] = useState("");
  const [browserModels, setBrowserModels] = useState<ModelInfo[]>([]);
  const [browserLive, setBrowserLive] = useState(false);
  const [browserLoading, setBrowserLoading] = useState(false);
  const [browserCustom, setBrowserCustom] = useState("");
  const [catalog, setCatalog] = useState<Record<string, ModelInfo[]>>({});

  const [provName, setProvName] = useState("");
  const [provKind, setProvKind] = useState("openai");
  const [provBase, setProvBase] = useState("");
  const [provModel, setProvModel] = useState("");
  const [provKey, setProvKey] = useState("");
  const [provResult, setProvResult] = useState<string | null>(null);

  const [diags, setDiags] = useState<DiagRow[]>([]);
  const [diagsRunning, setDiagsRunning] = useState(false);

  const streamTask = useRef<string | null>(null);
  const activitySeq = useRef(0);
  const recRef = useRef<{ stop: () => void } | null>(null);
  const micTimer = useRef<number | null>(null);

  const activeSession = sessions.find((s) => s.id === activeId) ?? null;

  const pushActivity = useCallback((kind: ActivityItem["kind"], text: string) => {
    setActivity((a) => [...a.slice(-60), { id: activitySeq.current++, kind, text }]);
  }, []);

  const loadSessions = useCallback(async () => {
    setSessions(await api.listSessions());
  }, []);

  const loadProviders = useCallback(async () => {
    const rows = await api.listProviders();
    setProviders(rows);
    const map: Record<string, boolean> = {};
    for (const p of rows) {
      try {
        map[p.name] = await api.hasProviderKey(p.name);
      } catch {
        map[p.name] = false;
      }
    }
    setKeys(map);
  }, []);

  useEffect(() => {
    void loadSessions();
    void loadProviders();
    void api.getSetting("agent_mode").then((m) => {
      if (m === "plan" || m === "build") setModeState(m);
    }).catch(() => undefined);
    const unlisten = onAgentEvent((ev) => {
      switch (ev.type) {
        case "task_created":
          streamTask.current = ev.task_id;
          setStreaming(true);
          break;
        case "model_delta":
          if (ev.task_id === streamTask.current) {
            setStream((s) => s + ev.text);
          }
          break;
        case "reasoning_delta":
          if (ev.task_id === streamTask.current) {
            setStreamReasoning((s) => s + ev.text);
          }
          break;
        case "message_completed":
          setMessages((m) => [
            ...m,
            {
              id: ev.task_id ?? crypto.randomUUID(),
              role: "assistant",
              content: ev.text,
              reasoning: ev.reasoning ?? "",
            },
          ]);
          setStream("");
          setStreamReasoning("");
          break;
        case "tool_requested":
          pushActivity("tool", `→ ${ev.tool}(${JSON.stringify(ev.args ?? {})})`);
          break;
        case "tool_output": {
          const out = String(ev.output ?? "").slice(0, 300);
          pushActivity("output", `${ev.tool}: ${out}${out.length >= 300 ? "…" : ""}`);
          break;
        }
        case "approval_requested":
          setApprovals((a) => [
            ...a,
            {
              approval_id: String(ev.approval_id),
              task_id: String(ev.task_id ?? ""),
              tool: String(ev.tool),
              args: ev.args,
              risk: String(ev.risk),
              reason: String(ev.reason),
            },
          ]);
          break;
        case "approval_resolved":
          setApprovals((a) => a.filter((x) => x.approval_id !== String(ev.approval_id)));
          break;
        case "task_completed": {
          const s = ev.summary;
          pushActivity(
            "ok",
            `✓ done · ${s.turns} turns · ${s.input_tokens}+${s.output_tokens} tokens`,
          );
          setLastSummary(s);
          setStreaming(false);
          break;
        }
        case "task_failed":
          pushActivity("err", `✗ ${String(ev.error ?? "task failed")}`);
          setStreaming(false);
          setMessages((m) => [
            ...m,
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `I ran into a problem and stopped:\n\n${String(ev.error ?? "unknown error")}`,
              reasoning: "",
            },
          ]);
          break;
        default:
          break;
      }
    });
    return () => {
      void unlisten.then((f) => f());
    };
  }, [loadSessions, loadProviders, pushActivity]);

  const openSession = useCallback(
    async (s: SessionRow) => {
      setActiveId(s.id);
      setStream("");
      setStreamReasoning("");
      setStreaming(false);
      setApprovals([]);
      setActivity([]);
      setMessages([]);
      setNotice(null);
      setLastSummary(null);
      setProviderSel(s.provider_name ?? "");
      setModelSel(s.model ?? "");
      setEffortSel(s.reasoning_effort ?? null);
      const rows = await api.getMessages(s.id);
      setMessages(
        rows
          .filter((r) => r.role === "user" || r.role === "assistant")
          .map((r) => ({
            id: r.id,
            role: r.role as "user" | "assistant",
            content: r.content,
            reasoning: r.reasoning ?? "",
          })),
      );
    },
    [],
  );

  const createSession = async () => {
    const s = await api.createSession("New session", newWorkspace.trim() || null);
    setNewWorkspace("");
    await loadSessions();
    await openSession(s);
  };

  const applyBinding = async (provider: string, model: string) => {
    if (!activeId || !provider || !model.trim()) return;
    try {
      await api.bindSessionProvider(activeId, provider, model.trim());
      setNotice(`Now using ${provider} / ${model.trim()} — history kept.`);
      await loadSessions();
    } catch (e) {
      setNotice(String(e));
    }
  };

  const send = async () => {
    if (!activeId) return;
    if (!draft.trim()) {
      setNotice("Type a message first.");
      return;
    }
    if (streaming) {
      setNotice("The agent is still working — allow or deny its pending approval, or wait.");
      return;
    }
    const text = draft.trim();
    setDraft("");
    setMessages((m) => [...m, { id: crypto.randomUUID(), role: "user", content: text, reasoning: "" }]);
    try {
      await api.sendMessage(activeId, text);
    } catch (e) {
      pushActivity("err", String(e));
      setNotice(String(e));
      setStreaming(false);
    }
  };

  const setMode = async (m: "build" | "plan") => {
    try {
      const msg = await api.setMode(m);
      setModeState(m);
      setNotice(msg);
    } catch (e) {
      setNotice(String(e));
    }
  };

  const setEffort = async (v: string | null) => {
    setEffortSel(v);
    if (!activeId) return;
    try {
      await api.setSessionEffort(activeId, v);
    } catch (e) {
      setNotice(String(e));
    }
  };

  const setAutonomy = async (v: string) => {
    setAutonomySel(v);
    try {
      await api.setAutonomy(v);
    } catch (e) {
      setNotice(String(e));
    }
  };

  const answerApproval = async (card: ApprovalCard, allow: boolean) => {
    setApprovals((a) => a.filter((x) => x.approval_id !== card.approval_id));
    try {
      await api.approve(card.approval_id, allow);
    } catch (e) {
      pushActivity("err", String(e));
    }
  };

  const openModels = async (provider: string) => {
    const name = provider || providerSel || providers[0]?.name || "";
    if (!name) {
      setNotice("Add a provider first.");
      return;
    }
    setBrowserProvider(name);
    setBrowserModels([]);
    setBrowserLive(false);
    setBrowserLoading(true);
    setBrowserCustom("");
    setShowModels(true);
    try {
      setBrowserModels(await api.liveModels(name));
      setBrowserLive(true);
    } catch {
      try {
        setBrowserModels(await api.knownModels(providers.find((p) => p.name === name)?.kind ?? ""));
      } catch (e) {
        setNotice(String(e));
      }
    } finally {
      setBrowserLoading(false);
    }
  };

  const pickModel = async (id: string) => {
    setModelSel(id);
    setShowModels(false);
    if (providerSel || browserProvider) {
      const p = providerSel || browserProvider;
      if (providerSel !== p) setProviderSel(p);
      await applyBinding(p, id);
    }
  };

  const runDiags = async () => {
    setDiags([]);
    setDiagsRunning(true);
    setShowDiags(true);
    try {
      setDiags(await api.runDiagnostics(activeId));
    } catch (e) {
      setDiags([{ name: "Runner", ok: false, detail: String(e) }]);
    } finally {
      setDiagsRunning(false);
    }
  };

  const stopMic = () => {
    recRef.current?.stop();
    recRef.current = null;
    if (micTimer.current !== null) {
      window.clearTimeout(micTimer.current);
      micTimer.current = null;
    }
    setMicOn(false);
  };

  const toggleMic = () => {
    if (micOn) {
      stopMic();
      return;
    }
    const w = window as unknown as Record<string, unknown>;
    const Ctor = (w.SpeechRecognition ?? w.webkitSpeechRecognition) as
      | (new () => {
          lang: string;
          interimResults: boolean;
          onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null;
          onerror: ((e: { error: string }) => void) | null;
          onend: (() => void) | null;
          start: () => void;
          stop: () => void;
        })
      | undefined;
    if (!Ctor) {
      setNotice("Voice input isn't supported in this WebView — type instead.");
      return;
    }
    try {
      const rec = new Ctor();
      rec.lang = "en-US";
      rec.interimResults = false;
      rec.onresult = (e) => {
        let text = "";
        for (let i = 0; i < e.results.length; i++) {
          const alt = e.results[i][0];
          if (alt) text += alt.transcript;
        }
        text = text.trim();
        if (text) {
          setDraft((d) => (d && !/\s$/.test(d) ? d + " " + text : d + text));
        }
      };
      rec.onerror = (e) => {
        setNotice(`Microphone error: ${e.error}`);
        stopMic();
      };
      rec.onend = () => {
        setMicOn(false);
        recRef.current = null;
      };
      rec.start();
      recRef.current = { stop: () => rec.stop() };
      setMicOn(true);
      micTimer.current = window.setTimeout(() => {
        stopMic();
        setNotice("Stopped listening (2 minute cap).");
      }, 120_000);
    } catch (e) {
      setNotice(String(e));
    }
  };

  const selectedKind = providers.find((p) => p.name === providerSel)?.kind ?? "openai";

  const modelDisplay = (() => {
    const all = [...(catalog[selectedKind] ?? []), ...browserModels];
    return all.find((m) => m.id === modelSel)?.display_name ?? modelSel;
  })();

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">✦</span> Super-AI <small>v0.2.0</small>
        </div>

        <div className="new-session">
          <input
            value={newWorkspace}
            onChange={(e) => setNewWorkspace(e.target.value)}
            placeholder="Folder for files (blank = Super-AI folder)"
            spellCheck={false}
          />
          <button className="primary" onClick={() => void createSession()}>
            + New session
          </button>
        </div>

        <div className="sessions">
          {sessions.map((s) => (
            <div key={s.id} className="session-row">
              <button
                className={`session ${s.id === activeId ? "active" : ""}`}
                onClick={() => void openSession(s)}
              >
                <span className="session-title">{s.title}</span>
                <span className="session-meta">
                  {s.provider_name ?? "no provider"} · {s.model ?? "no model"}
                </span>
              </button>
              <button
                className="icon-btn danger"
                title="Delete session"
                onClick={() => {
                  void api
                    .deleteSession(s.id)
                    .then(() => loadSessions())
                    .then(() => {
                      if (s.id === activeId) {
                        setActiveId(null);
                        setMessages([]);
                      }
                    })
                    .catch((e: unknown) => setNotice(String(e)));
                }}
              >
                ✕
              </button>
            </div>
          ))}
          {sessions.length === 0 && <div className="hint">No sessions yet — create one above.</div>}
        </div>

        <div className="controls">
          <div className="control-row">
            <label>Autonomy</label>
            <select value={autonomy} onChange={(e) => void setAutonomy(e.target.value)}>
              <option value="low">Low — ask me</option>
              <option value="medium">Medium</option>
              <option value="high">High — auto</option>
            </select>
          </div>
          <div className="control-row">
            <label>Reasoning</label>
            <select
              value={effortSel ?? ""}
              disabled={!activeId}
              onChange={(e) => void setEffort(e.target.value || null)}
            >
              <option value="">Default</option>
              {EFFORTS.map((v) => (
                <option key={v} value={v}>
                  {EFFORT_LABELS[v]}
                </option>
              ))}
            </select>
          </div>
          <div className="hint">How hard this session's model thinks.</div>
        </div>

        <div className="sidebar-footer">
          <button onClick={() => void runDiags()}>🩺 Diagnostics</button>
          <button onClick={() => setShowProviders(true)}>⚙ Providers &amp; keys</button>
          <div className="version">BYOK · keys stay in Credential Manager</div>
        </div>
      </aside>

      <main className="main">
        <header className="chat-header">
          <div className="chat-title">
            {activeSession?.title ?? "Super-AI"}
            {activeSession?.workspace && (
              <span className="workspace-tag">{activeSession.workspace}</span>
            )}
            {lastSummary && (
              <span className="usage-chip">
                {lastSummary.turns} turns · {lastSummary.input_tokens}+{lastSummary.output_tokens} tok
              </span>
            )}
          </div>
          {activeSession && (
            <div className="binding">
              <select value={providerSel} onChange={(e) => {
                const name = e.target.value;
                setProviderSel(name);
                if (name && modelSel) void applyBinding(name, modelSel);
              }}>
                <option value="">provider…</option>
                {providers.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name} ({KINDS[p.kind]?.label ?? p.kind})
                  </option>
                ))}
              </select>
              <button
                className="model-btn"
                title="Pick a model — switches instantly, history kept"
                onClick={() => void openModels(providerSel)}
              >
                {modelSel
                  ? modelDisplay
                  : "Choose model…"}
              </button>
            </div>
          )}
          <div className="mode-strip">
            <button
              className={`mode-pill build ${mode === "build" ? "on" : ""}`}
              title="Build mode: the agent can act (approvals still apply)"
              onClick={() => void setMode("build")}
            >
              Build
            </button>
            <button
              className={`mode-pill plan ${mode === "plan" ? "on" : ""}`}
              title="Plan mode: read-only, the agent plans but changes nothing"
              onClick={() => void setMode("plan")}
            >
              Plan
            </button>
            <span className="mode-hint">
              {mode === "build" ? "acts with approval prompts" : "read-only — Tab to switch"}
            </span>
          </div>
        </header>

        {approvals.length > 0 && (
          <div className="approval-banner">
            ⏳ {approvals.length} action{approvals.length === 1 ? "" : "s"} need
            {approvals.length === 1 ? "s" : ""} your approval to continue — see below
          </div>
        )}

        <div className="messages">
          {!activeSession && sessions.length === 0 && (
            <div className="empty-hero">
              <h2>Welcome to Super-AI</h2>
              <p>Your AI assistant for files and terminal tasks. Three steps:</p>
              <div className="steps">
                <div className="step">
                  <b>1. Add a provider</b>
                  <p>ChatGPT, Claude, Grok, Mistral, Gemini — or free local models with Ollama.</p>
                  <div>
                    <button onClick={() => setShowProviders(true)}>Open Providers &amp; keys</button>
                  </div>
                </div>
                <div className="step">
                  <b>2. Create a session</b>
                  <p>Type a folder path left (or leave blank for your Super-AI folder), then New session.</p>
                  <div>
                    <button onClick={() => void createSession()}>New session</button>
                  </div>
                </div>
                <div className="step">
                  <b>3. Choose a model and chat</b>
                  <p>Pick provider + model above — changeable any time, even mid-conversation.</p>
                </div>
              </div>
            </div>
          )}
          {messages.map((m) => (
            <div key={m.id} className={`msg ${m.role}`}>
              <div className="msg-head">{m.role === "user" ? "you" : "assistant"}</div>
              {m.reasoning && (
                <details className="thinking">
                  <summary>Thinking</summary>
                  <div className="thinking-body">{m.reasoning}</div>
                </details>
              )}
              <div className="msg-content">
                <Markdown remarkPlugins={[remarkGfm]}>{m.content}</Markdown>
              </div>
            </div>
          ))}
          {(stream || streamReasoning) && (
            <div className="msg assistant streaming">
              <div className="msg-head">
                assistant · LIVE · {(stream + streamReasoning).length} chars
              </div>
              {streamReasoning && (
                <details className="thinking" open>
                  <summary>Thinking</summary>
                  <div className="thinking-body">{streamReasoning}</div>
                </details>
              )}
              {stream && (
                <div className="msg-content">
                  <Markdown remarkPlugins={[remarkGfm]}>{stream + "▍"}</Markdown>
                </div>
              )}
            </div>
          )}
          {activeSession && messages.length === 0 && !stream && (
            <div className="empty-hero">
              <p>Pick a provider and press “Choose model…” above if you haven’t yet — then ask for something.</p>
            </div>
          )}
        </div>

        {approvals.length > 0 && (
          <div className="approvals">
            {approvals.map((card) => (
              <div key={card.approval_id} className="approval-card">
                <div className="approval-head">
                  <span className={`risk risk-${card.risk}`}>{card.risk}</span>
                  <span className="approval-tool">{card.tool}</span>
                </div>
                <pre className="approval-args">{JSON.stringify(card.args, null, 2)}</pre>
                <div className="approval-reason">{card.reason}</div>
                <div className="approval-actions">
                  <button className="btn-allow" onClick={() => void answerApproval(card, true)}>
                    Allow once
                  </button>
                  <button className="btn-deny" onClick={() => void answerApproval(card, false)}>
                    Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {activity.length > 0 && (
          <div className="activity">
            {activity
              .slice(-20)
              .reverse()
              .map((a) => (
                <div key={a.id} className={`activity-item ${a.kind}`}>
                  {a.text}
                </div>
              ))}
          </div>
        )}

        <div className="composer">
          {notice && <div className="notice">{notice}</div>}
          <div className="composer-row">
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                } else if (e.key === "Tab") {
                  e.preventDefault();
                  void setMode(mode === "build" ? "plan" : "build");
                }
              }}
              placeholder={
                streaming
                  ? "Agent is working…"
                  : activeSession?.provider_name
                    ? "Ask the agent something… (Enter sends · Tab switches mode)"
                    : "Bind a provider and model above first…"
              }
              rows={3}
            />
            <button
              className={`mic-btn ${micOn ? "recording" : ""}`}
              title={micOn ? "Stop and transcribe" : "Dictate with microphone"}
              onClick={toggleMic}
            >
              {micOn ? "⏹" : "🎙"}
            </button>
            <button className="send primary" onClick={() => void send()}>
              Send
            </button>
          </div>
        </div>
      </main>

      {showProviders && (
        <ProvidersModal
          providers={providers}
          keys={keys}
          catalog={catalog}
          setCatalog={setCatalog}
          onClose={() => setShowProviders(false)}
          onChanged={() => void loadProviders()}
          initialName={provName}
          setInitialName={setProvName}
          initialKind={provKind}
          setInitialKind={setProvKind}
          initialBase={provBase}
          setInitialBase={setProvBase}
          initialModel={provModel}
          setInitialModel={setProvModel}
          apiKey={provKey}
          setApiKey={setProvKey}
          result={provResult}
          setResult={setProvResult}
        />
      )}

      {showModels && (
        <div className="modal-backdrop" onClick={() => setShowModels(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Choose a model</h2>
              <button className="close" onClick={() => setShowModels(false)}>
                ×
              </button>
            </div>
            <p className="modal-note">
              Friendly names, prices and context. {browserLive ? "Live list from your server." : "Built-in catalog."}{" "}
              Switching is instant — history kept, next answer uses the new model.
            </p>
            <div className="form-row">
              <label>Provider</label>
              <select
                value={browserProvider}
                onChange={(e) => {
                  const name = e.target.value;
                  setBrowserProvider(name);
                  setBrowserLoading(true);
                  setBrowserModels([]);
                  void api
                    .liveModels(name)
                    .then((rows) => {
                      setBrowserModels(rows);
                      setBrowserLive(true);
                    })
                    .catch(() => {
                      const kind = providers.find((p) => p.name === name)?.kind ?? "";
                      void api.knownModels(kind).then(setBrowserModels).catch(() => undefined);
                    })
                    .finally(() => setBrowserLoading(false));
                }}
              >
                {providers.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  setBrowserLoading(true);
                  void openModels(browserProvider);
                }}
              >
                Reload
              </button>
            </div>
            <div className="form-row">
              <label>Custom</label>
              <input
                value={browserCustom}
                onChange={(e) => setBrowserCustom(e.target.value)}
                placeholder="Or type any model id…"
                spellCheck={false}
              />
              <button
                onClick={() => {
                  if (browserCustom.trim()) void pickModel(browserCustom.trim());
                }}
              >
                Use
              </button>
            </div>
            {browserLoading && <div className="hint">Loading live model list…</div>}
            <div style={{ marginTop: 10 }}>
              {browserModels.map((m) => (
                <button key={m.id} className="model-row" onClick={() => void pickModel(m.id)}>
                  <b>{m.display_name}</b>
                  <span className="cid">{m.id}</span>
                  <span className="cmeta">{catalogLine(m)}</span>
                </button>
              ))}
              {!browserLoading && browserModels.length === 0 && (
                <div className="hint">No models found — check the provider setup.</div>
              )}
            </div>
          </div>
        </div>
      )}

      {showDiags && (
        <div className="modal-backdrop" onClick={() => setShowDiags(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h2>Diagnostics</h2>
              <button className="close" onClick={() => setShowDiags(false)}>
                ×
              </button>
            </div>
            <p className="modal-note">
              Self-test for the whole pipeline. No tokens spent, nothing modified.
            </p>
            {diagsRunning && diags.length === 0 && <div className="hint">Running checks…</div>}
            {diags.map((d) => (
              <div key={d.name} className="diag-row">
                <div className="diag-head">
                  <span className={`dot ${d.ok === true ? "ok" : d.ok === false ? "fail" : "skip"}`} />
                  {d.name}
                </div>
                <div className="diag-detail">{d.detail}</div>
              </div>
            ))}
            <div className="form-row" style={{ marginTop: 12 }}>
              <button onClick={() => void runDiags()}>Re-run</button>
              <button onClick={() => setShowDiags(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ProvidersModal(props: {
    providers: ProviderRow[];
    keys: Record<string, boolean>;
    catalog: Record<string, ModelInfo[]>;
    setCatalog: (c: Record<string, ModelInfo[]>) => void;
    onClose: () => void;
    onChanged: () => void;
    initialName: string;
    setInitialName: (v: string) => void;
    initialKind: string;
    setInitialKind: (v: string) => void;
    initialBase: string;
    setInitialBase: (v: string) => void;
    initialModel: string;
    setInitialModel: (v: string) => void;
    apiKey: string;
    setApiKey: (v: string) => void;
    result: string | null;
    setResult: (v: string | null) => void;
  }) {
    const {
      providers: plist,
      keys: kmap,
      catalog: cat,
      setCatalog: setCat,
      onClose,
      onChanged,
      initialName: name,
      setInitialName: setName,
      initialKind: kind,
      setInitialKind: setKind,
      initialBase: base,
      setInitialBase: setBase,
      initialModel: dmodel,
      setInitialModel: setDmodel,
      apiKey: key,
      setApiKey: setKey,
      result,
      setResult,
    } = props;
    const meta = KINDS[kind] ?? KINDS.openai;

    const ensureCatalog = (k: string) => {
      if (cat[k]) return;
      void api
        .knownModels(k)
        .then((rows) => setCat({ ...cat, [k]: rows }))
        .catch(() => undefined);
    };

    const save = async () => {
      setResult(null);
      try {
        const msg = await api.saveProvider(
          name,
          kind,
          base.trim() || null,
          dmodel.trim() || null,
          key.trim() || null,
        );
        setResult(msg);
        setKey("");
        onChanged();
      } catch (e) {
        setResult(String(e));
      }
    };

    const test = async (providerName: string) => {
      try {
        setResult(await api.testProvider(providerName));
      } catch (e) {
        setResult(String(e));
      }
    };

    return (
      <div className="modal-backdrop" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <div className="modal-head">
            <h2>Providers &amp; API keys</h2>
            <button className="close" onClick={onClose}>
              ×
            </button>
          </div>
          <p className="modal-note">
            Keys live in Windows Credential Manager — never in the database or logs. Ollama /
            local servers need no key.
          </p>

          {plist.map((p) => (
            <div key={p.name} className="provider-row">
              <div className="provider-info">
                <strong>{p.name}</strong>
                <span>
                  {KINDS[p.kind]?.label ?? p.kind} · {p.default_model ?? "no default model"}
                </span>
                <span className={`key-status ${kmap[p.name] ? "ok" : "missing"}`}>
                  {kmap[p.name] ? "✓ key stored" : "⚠ no key"}
                </span>
              </div>
              <div className="row-btns">
                <button onClick={() => void test(p.name)}>Test</button>
                <button
                  className="danger"
                  onClick={() => {
                    void api
                      .deleteProvider(p.name)
                      .then(() => onChanged())
                      .catch((e: unknown) => setResult(String(e)));
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
          {plist.length === 0 && <div className="hint">No providers yet — add one below.</div>}

          <div className="provider-form">
            <h3 style={{ margin: "6px 0 0" }}>Add / update provider</h3>
            <div className="form-row">
              <label>Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. my-mistral"
              />
              <label>Kind</label>
              <select value={kind} onChange={(e) => setKind(e.target.value)}>
                {KIND_IDS.map((k) => (
                  <option key={k} value={k}>
                    {KINDS[k].label}
                  </option>
                ))}
              </select>
            </div>
            <div className="hint">{meta.blurb}</div>
            <div className="form-row">
              <label>Base URL</label>
              <input
                value={base}
                onChange={(e) => setBase(e.target.value)}
                placeholder={meta.baseHint}
                spellCheck={false}
              />
              {meta.base && <button onClick={() => setBase(meta.base ?? "")}>Fill default</button>}
            </div>
            <div className="form-row">
              <label>Model</label>
              <input
                value={dmodel}
                onChange={(e) => setDmodel(e.target.value)}
                placeholder="default model id"
                spellCheck={false}
              />
            </div>
            <div className="chips">
              <span className="lbl">try:</span>
              {meta.hints.map((h) => (
                <button key={h} onClick={() => setDmodel(h)}>
                  {h}
                </button>
              ))}
            </div>
            <div className="catalog">
              <button
                onClick={() => ensureCatalog(kind)}
                title="Show built-in model catalog"
              >
                Model catalog
              </button>
              {(cat[kind] ?? []).map((m) => (
                <div key={m.id} className="catalog-item">
                  <div>
                    <strong>{m.display_name}</strong> <span className="cid">{m.id}</span>
                  </div>
                  <div className="cmeta">{catalogLine(m)}</div>
                </div>
              ))}
            </div>
            <div className="form-row">
              <label>API key</label>
              <input
                type="password"
                value={key}
                onChange={(e) => setKey(e.target.value)}
                placeholder={
                  kind === "ollama" || kind === "local"
                    ? "optional for local servers"
                    : "required (leave empty to keep existing)"
                }
              />
              <button className="primary" onClick={() => void save()}>
                Save provider
              </button>
            </div>
          </div>

          {result && <div className="test-result">{result}</div>}
        </div>
      </div>
  );
}
