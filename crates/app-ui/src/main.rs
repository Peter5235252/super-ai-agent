//! Super-AI native desktop app (100% Rust, no web shell).
//!
//! Single `super-ai-native` binary: eframe/egui immediate-mode UI talking
//! directly to the agent runtime — no WebView, no dev server, no IPC
//! bridge. Sessions/messages/providers live in SQLite, keys in the OS
//! credential store, and every agent step flows through the broadcast
//! event bus (UI feed + flight-recorder persistence, same as before).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![forbid(unsafe_code)]

mod md;
mod stt;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use agent_runtime::{Agent, AgentEvent, TaskRequest, TaskSummary, ToolRegistry};
use app_core::SYSTEM_PROMPT;
use app_core::{AgentMode, ApprovalDecision, RiskClass, SideEffect, WorkspaceConfig, now_ms};
use futures::StreamExt;
use lucide_icons::Icon;
use persistence::{Db, MessageRow, ProviderRow, SessionRow};
use policy_engine::{Decision, PolicyEngine};
use provider_api::ReasoningEffort;
use provider_api::{Message, MessageRole, ModelInfo, ModelProvider};
use secrecy::SecretString;
use tokio::sync::broadcast;
use tool_core::ToolContext;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

const SERVICE: &str = "super-ai";
const APP_ID: &str = "com.superai.desktop";

use app_core::SYSTEM_PROMPT;

/// Provider kinds the shell can construct: cloud (key required) + local.
const KINDS: &[&str] = &[
    "openai",
    "anthropic",
    "xai",
    "mistral",
    "gemini",
    "ollama",
    "local",
];

fn model_hints(kind: &str) -> &'static [&'static str] {
    match kind {
        "openai" => &[
            "gpt-6-astra",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.6-cyber",
        ],
        "anthropic" => &[
            "claude-fable-5-1",
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-haiku-4-5",
        ],
        "xai" => &["grok-4.6", "grok-4.5", "grok-4.3", "grok-build-0.1"],
        "mistral" => &[
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "codestral-latest",
            "devstral-latest",
        ],
        "gemini" => &[
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.6-flash",
        ],
        "ollama" => &[
            "llama4:scout",
            "qwen3:14b",
            "qwen3:8b",
            "llama3.1:8b",
            "mistral:7b",
            "llama3.2:3b",
        ],
        _ => &[],
    }
}

fn base_placeholder(kind: &str) -> &'static str {
    match kind {
        "openai" => "https://api.openai.com/v1",
        "mistral" => "https://api.mistral.ai/v1",
        "gemini" => "https://generativelanguage.googleapis.com/v1beta/openai",
        "ollama" => "http://localhost:11434/v1",
        "local" => "http://localhost:1234/v1 (LM Studio) or :8000 (vLLM)",
        _ => "(provider default)",
    }
}

fn key_required(kind: &str) -> bool {
    matches!(kind, "openai" | "anthropic" | "xai" | "mistral" | "gemini")
}

/// Display name for a stored provider-kind id.
fn kind_label(kind: &str) -> &'static str {
    match kind {
        "openai" => "OpenAI",
        "anthropic" => "Anthropic",
        "xai" => "SpaceXAI",
        "mistral" => "Mistral",
        "gemini" => "Gemini",
        "ollama" => "Ollama",
        "local" => "Local",
        _ => "Custom",
    }
}

/// Plain-language blurbs so non-technical users can pick a kind.
fn kind_blurb(kind: &str) -> &'static str {
    match kind {
        "openai" => "ChatGPT models. Needs an OpenAI API key.",
        "anthropic" => "Claude models. Needs an Anthropic API key.",
        "xai" => "Grok models by SpaceXAI. Needs an xAI API key.",
        "mistral" => "Mistral + Codestral. Needs a La Plateforme API key.",
        "gemini" => "Google Gemini. Needs a Google AI Studio key.",
        "ollama" => {
            "Free models on your own PC via Ollama. No key — install Ollama and pull a model first."
        }
        "local" => "LM Studio, vLLM or llama.cpp server. No key — enter its address below.",
        _ => "",
    }
}

/// Default server address, if the kind has one well-known value.
fn default_base_url(kind: &str) -> Option<&'static str> {
    match kind {
        "mistral" => Some("https://api.mistral.ai/v1"),
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta/openai"),
        "ollama" => Some("http://localhost:11434/v1"),
        "local" => Some("http://localhost:1234/v1"),
        _ => None,
    }
}

/// One Lucide glyph in the bundled icon font.
fn li(ic: Icon) -> egui::RichText {
    egui::RichText::new(char::from(ic).to_string()).family(egui::FontFamily::Name("lucide".into()))
}

/// Font setup shared by the app and the glyph-coverage test.
///
/// The `lucide` family lists `Hack` second: an icons-only font has no `?`
/// glyph, and without a fallback epaint logs a "replacement character"
/// warning and renders missing glyphs blank. With the fallback, any icon
/// missing from the bundled font shows a visible `?` instead.
fn app_fonts() -> egui::FontDefinitions {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "lucide".to_owned(),
        egui::FontData::from_static(lucide_icons::LUCIDE_FONT_BYTES).into(),
    );
    fonts.families.insert(
        egui::FontFamily::Name("lucide".into()),
        vec!["lucide".to_owned(), "Hack".to_owned()],
    );
    fonts
}

// ---------------------------------------------------------------------------
// Background -> UI channel
// ---------------------------------------------------------------------------

enum UiUpdate {
    Notice(String),
    Sessions(Vec<SessionRow>),
    SessionCreated(SessionRow),
    SessionMessages {
        session: Uuid,
        messages: Vec<MessageRow>,
    },
    Providers(Vec<ProviderRow>),
    Keys(HashMap<String, bool>),
    ProviderTested(String),
    Agent(AgentEvent),
    SendFailed(String),
    SessionGone(Uuid),
    ModelList {
        provider: String,
        models: Vec<ModelInfo>,
    },
    /// Speech-to-text finished: transcribed draft text, or a plain error.
    SttDone(Result<String, String>),
    /// Self-test results for the diagnostics modal.
    Diagnostics(Vec<DiagRow>),
}

/// One self-test row: `ok` is None for "skipped".
#[derive(Clone)]
struct DiagRow {
    name: String,
    ok: Option<bool>,
    detail: String,
}

// ---------------------------------------------------------------------------
// View state
// ---------------------------------------------------------------------------

struct UiMsg {
    role: &'static str,
    content: String,
    /// The model's private reasoning for assistant turns (may be empty).
    reasoning: String,
}

struct ActivityItem {
    kind: &'static str,
    text: String,
}

struct ApprovalCard {
    approval_id: Uuid,
    tool: String,
    args: serde_json::Value,
    risk: String,
    reason: String,
}

struct SuperAiApp {
    db: Db,
    policy: Arc<PolicyEngine>,
    agent: Arc<Agent>,
    rt: tokio::runtime::Handle,
    tx: mpsc::Sender<UiUpdate>,
    rx: mpsc::Receiver<UiUpdate>,

    sessions: Vec<SessionRow>,
    active_id: Option<Uuid>,
    messages: Vec<UiMsg>,
    stream: String,
    stream_task: Option<Uuid>,
    streaming: bool,
    /// Live chain-of-thought for the streaming turn (visible "Thinking").
    stream_reasoning: String,
    stream_reasoning_task: Option<Uuid>,
    /// Last time any agent event arrived; guards against a stuck spinner
    /// if a terminal event is ever lost.
    last_agent_event: Option<std::time::Instant>,
    activity: Vec<ActivityItem>,
    approvals: Vec<ApprovalCard>,
    providers: Vec<ProviderRow>,
    keys: HashMap<String, bool>,
    notice: Option<String>,
    last_summary: Option<TaskSummary>,
    autonomy: RiskClass,
    /// Per-session reasoning effort override (None = provider default).
    effort_sel: Option<ReasoningEffort>,
    /// Mirrors the policy engine's operating mode (Plan/Build).
    mode_mirror: AgentMode,
    /// Last ESC keypress, for the double-ESC stop gesture.
    last_esc: Option<std::time::Instant>,
    /// Live microphone capture (Some while recording).
    stt: Option<stt::Recorder>,
    /// A transcription/download is running in the background.
    stt_working: bool,
    /// Focus the composer on the next frame (e.g. after opening a session).
    focus_composer: bool,
    /// A binding change is being saved (guards the seamless auto-rebind).
    rebinding: bool,
    /// Diagnostics modal + last results + running flag.
    show_diags: bool,
    diags: Vec<DiagRow>,
    diags_running: bool,

    show_providers: bool,
    new_workspace: String,
    draft: String,
    provider_sel: String,
    model_sel: String,
    send_requested: bool,
    /// Guided model browser (friendly names, prices, live list).
    show_models: bool,
    browser_provider: String,
    browser_models: Vec<ModelInfo>,
    browser_loading: bool,
    browser_custom: String,

    prov_name: String,
    prov_kind: String,
    prov_base: String,
    prov_model: String,
    prov_key: String,
    prov_result: Option<String>,
}

impl SuperAiApp {
    fn new(
        db: Db,
        policy: Arc<PolicyEngine>,
        agent: Arc<Agent>,
        rt: tokio::runtime::Handle,
        tx: mpsc::Sender<UiUpdate>,
        rx: mpsc::Receiver<UiUpdate>,
        mode: AgentMode,
    ) -> Self {
        let app = Self {
            db,
            policy,
            agent,
            rt,
            tx,
            rx,
            sessions: Vec::new(),
            active_id: None,
            messages: Vec::new(),
            stream: String::new(),
            stream_task: None,
            streaming: false,
            stream_reasoning: String::new(),
            stream_reasoning_task: None,
            last_agent_event: None,
            activity: Vec::new(),
            approvals: Vec::new(),
            providers: Vec::new(),
            keys: HashMap::new(),
            notice: None,
            last_summary: None,
            autonomy: RiskClass::Low,
            effort_sel: None,
            mode_mirror: mode,
            last_esc: None,
            stt: None,
            stt_working: false,
            focus_composer: false,
            rebinding: false,
            show_diags: false,
            diags: Vec::new(),
            diags_running: false,
            show_providers: false,
            new_workspace: String::new(),
            draft: String::new(),
            provider_sel: String::new(),
            model_sel: String::new(),
            send_requested: false,
            show_models: false,
            browser_provider: String::new(),
            browser_models: Vec::new(),
            browser_loading: false,
            browser_custom: String::new(),
            prov_name: String::new(),
            prov_kind: "openai".to_string(),
            prov_base: String::new(),
            prov_model: String::new(),
            prov_key: String::new(),
            prov_result: None,
        };
        app.refresh_sessions();
        app.refresh_providers();
        app
    }

    fn spawn(&self, fut: impl std::future::Future<Output = ()> + Send + 'static) {
        self.rt.spawn(fut);
    }

    fn push_activity(&mut self, kind: &'static str, text: String) {
        self.activity.push(ActivityItem { kind, text });
        if self.activity.len() > 60 {
            let excess = self.activity.len() - 60;
            self.activity.drain(..excess);
        }
    }

    // -- loaders ----------------------------------------------------------

    fn refresh_sessions(&self) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match db.list_sessions().await {
                Ok(rows) => {
                    let _ = tx.send(UiUpdate::Sessions(rows));
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("sessions: {e}")));
                }
            }
        });
    }

    fn refresh_providers(&self) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match db.list_providers().await {
                Ok(rows) => {
                    let mut keys = HashMap::new();
                    for p in &rows {
                        let has = secrets::SecretStore::get(SERVICE, &p.name)
                            .map(|o| o.is_some())
                            .unwrap_or(false);
                        keys.insert(p.name.clone(), has);
                    }
                    let _ = tx.send(UiUpdate::Providers(rows));
                    let _ = tx.send(UiUpdate::Keys(keys));
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("providers: {e}")));
                }
            }
        });
    }

    fn load_messages(&self, session: Uuid) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match db.list_messages(session).await {
                Ok(rows) => {
                    let _ = tx.send(UiUpdate::SessionMessages {
                        session,
                        messages: rows,
                    });
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("messages: {e}")));
                }
            }
        });
    }

    fn open_session(&mut self, id: Uuid) {
        self.active_id = Some(id);
        self.stream.clear();
        self.streaming = false;
        self.stream_task = None;
        self.stream_reasoning.clear();
        self.stream_reasoning_task = None;
        self.approvals.clear();
        self.activity.clear();
        self.messages.clear();
        self.notice = None;
        if let Some(s) = self.sessions.iter().find(|s| s.id == id) {
            self.provider_sel = s.provider_name.clone().unwrap_or_default();
            self.model_sel = s.model.clone().unwrap_or_default();
            self.effort_sel = s
                .reasoning_effort
                .as_deref()
                .and_then(ReasoningEffort::parse);
        }
        self.focus_composer = true;
        self.load_messages(id);
    }

    fn create_session(&mut self) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        let ws = self.new_workspace.trim().to_string();
        self.new_workspace.clear();
        self.spawn(async move {
            let id = Uuid::new_v4();
            // Blank input falls back to ~/Super-AI so file and terminal
            // tools work immediately instead of failing outright.
            let ws_path = if ws.is_empty() {
                let dir = default_workspace_dir();
                let _ = std::fs::create_dir_all(&dir);
                dir.to_string_lossy().to_string()
            } else {
                ws
            };
            match db
                .create_session(id, "New session", Some(ws_path.as_str()))
                .await
            {
                Ok(row) => {
                    let _ = tx.send(UiUpdate::SessionCreated(row));
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("new session: {e}")));
                }
            }
        });
    }

    fn delete_session(&mut self, id: Uuid) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match db.delete_session(id).await {
                Ok(()) => {
                    let _ = tx.send(UiUpdate::SessionGone(id));
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("delete session: {e}")));
                }
            }
        });
    }

    fn set_autonomy(&mut self, risk: RiskClass) {
        self.autonomy = risk;
        let policy = self.policy.clone();
        self.spawn(async move {
            policy.set_auto_approve(risk).await;
        });
    }

    fn set_mode(&mut self, mode: AgentMode) {
        self.mode_mirror = mode;
        let policy = self.policy.clone();
        let db = self.db.clone();
        self.spawn(async move {
            policy.set_mode(mode).await;
            let _ = db.set_setting("agent_mode", mode.as_str()).await;
        });
        self.notice = Some(if mode == AgentMode::Build {
            "Build mode — the agent can act (writes and commands still need your approval)."
                .to_string()
        } else {
            "Plan mode — read-only: the agent can look and explain, not change anything."
                .to_string()
        });
    }

    fn toggle_mode(&mut self) {
        self.set_mode(self.mode_mirror.toggle());
    }

    fn start_mic(&mut self) {
        if self.stt_working {
            return;
        }
        match stt::start_recording() {
            Ok(rec) => {
                self.notice = Some(format!(
                    "Listening on {} — click the mic again to transcribe.",
                    rec.device_name
                ));
                self.stt = Some(rec);
            }
            Err(e) => self.notice = Some(e),
        }
    }

    fn stop_and_transcribe(&mut self) {
        let Some(rec) = self.stt.take() else {
            return;
        };
        let recording = rec.finish();
        if recording.samples.is_empty() {
            self.notice = Some("Nothing recorded.".to_string());
            return;
        }
        self.stt_working = true;
        self.notice = Some("Transcribing… (first use downloads a 75 MB speech model)".to_string());
        let tx = self.tx.clone();
        self.spawn(async move {
            let pcm = stt::to_mono_16k(
                &recording.samples,
                recording.channels,
                recording.sample_rate,
            );
            if stt::peak(&pcm) < stt::SILENCE_PEAK {
                let _ = tx.send(UiUpdate::SttDone(Err(
                    "Too quiet — heard nothing. Try speaking closer to the mic.".to_string(),
                )));
                return;
            }
            let dir = data_dir();
            match stt::ensure_model(&dir).await {
                Ok(path) => {
                    let out = tokio::task::spawn_blocking(move || stt::transcribe(&path, &pcm))
                        .await
                        .map_err(|e| format!("Transcription crashed: {e}"))
                        .and_then(|r| r);
                    let _ = tx.send(UiUpdate::SttDone(out));
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::SttDone(Err(e)));
                }
            }
        });
    }

    /// Double-ESC stop gesture: first press arms, second press (within
    /// 1.5 s) aborts every agent task and clears pending approvals.
    fn handle_esc(&mut self) {
        let now = std::time::Instant::now();
        let armed = self
            .last_esc
            .map(|t| now.duration_since(t) < std::time::Duration::from_millis(1500))
            .unwrap_or(false);
        self.last_esc = Some(now);
        if !armed {
            self.notice = Some("Press ESC again to stop the agent dead.".to_string());
            return;
        }
        self.last_esc = None;
        let was_working = self.streaming || !self.approvals.is_empty();
        self.agent.abort_all();
        let policy = self.policy.clone();
        self.spawn(async move {
            policy.abort_pending().await;
        });
        self.streaming = false;
        self.stream.clear();
        self.stream_reasoning.clear();
        self.approvals.clear();
        if was_working {
            self.messages.push(UiMsg {
                role: "assistant",
                content: "Stopped. What would you like to do instead?".to_string(),
                reasoning: String::new(),
            });
            self.push_activity("err", "stopped by user (ESC x2)".to_string());
            self.notice = Some("Agent stopped.".to_string());
        } else {
            self.notice = Some("Nothing running.".to_string());
        }
    }

    // -- chat -------------------------------------------------------------

    fn queue_send(&mut self) {
        self.send_requested = true;
    }

    fn do_send(&mut self) {
        let Some(session_id) = self.active_id else {
            return;
        };
        let text = self.draft.trim().to_string();
        if text.is_empty() {
            return;
        }
        // Watchdog: if a previous task's terminal event was ever lost, the
        // spinner would stick forever and every send would be swallowed.
        // After 5 silent minutes, assume the task is gone and start fresh.
        if self.streaming {
            let silent = self
                .last_agent_event
                .map(|t| t.elapsed() > std::time::Duration::from_secs(300))
                .unwrap_or(true);
            if silent {
                self.streaming = false;
                self.stream.clear();
                self.stream_reasoning.clear();
                self.notice = Some(
                    "The previous task went quiet, so I cleared it. Trying your message now."
                        .to_string(),
                );
            } else {
                return;
            }
        }
        self.draft.clear();
        self.messages.push(UiMsg {
            role: "user",
            content: text.clone(),
            reasoning: String::new(),
        });

        let db = self.db.clone();
        let agent = self.agent.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match send_message_inner(&db, &agent, session_id, text).await {
                Ok(()) => {}
                Err(e) => {
                    let _ = tx.send(UiUpdate::SendFailed(e));
                }
            }
        });
    }

    fn apply_binding(&mut self) {
        let (Some(session_id), provider, model) = (
            self.active_id,
            self.provider_sel.trim().to_string(),
            self.model_sel.trim().to_string(),
        ) else {
            return;
        };
        if provider.is_empty() || model.is_empty() {
            return;
        }
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            match db.bind_provider(session_id, &provider, &model).await {
                Ok(()) => {
                    let _ = tx.send(UiUpdate::Notice(format!(
                        "Now using {provider} / {model} — history kept, applies to your next message"
                    )));
                    match db.list_sessions().await {
                        Ok(rows) => {
                            let _ = tx.send(UiUpdate::Sessions(rows));
                        }
                        Err(e) => {
                            let _ = tx.send(UiUpdate::Notice(format!("sessions: {e}")));
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::Notice(format!("bind: {e}")));
                }
            }
        });
    }

    // -- providers --------------------------------------------------------

    fn save_provider(&mut self) {
        let name = self.prov_name.trim().to_string();
        let kind = self.prov_kind.clone();
        let base = none_if_empty(self.prov_base.trim());
        let model = none_if_empty(self.prov_model.trim());
        let key = none_if_empty(self.prov_key.trim());
        if name.is_empty() {
            self.prov_result = Some("provider name is required".into());
            return;
        }
        if !name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
        {
            self.prov_result = Some("name may only contain letters, digits, '-', '_', '.'".into());
            return;
        }
        if !KINDS.contains(&kind.as_str()) {
            self.prov_result = Some(format!("unsupported provider kind: {kind}"));
            return;
        }
        if key_required(&kind) && key.is_none() {
            // Allow saving without a key (key can be added later), but say so.
        }
        self.prov_key.clear();
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            if let Some(k) = key.filter(|k| !k.is_empty()) {
                let secret = SecretString::new(k.into());
                if let Err(e) = secrets::SecretStore::set(SERVICE, &name, &secret) {
                    let _ = tx.send(UiUpdate::ProviderTested(format!("key store: {e}")));
                    return;
                }
            }
            let now = now_ms();
            let row = ProviderRow {
                name: name.clone(),
                kind,
                base_url: base,
                default_model: model,
                is_default: false,
                created_at: now,
                updated_at: now,
            };
            match db.upsert_provider(&row).await {
                Ok(()) => {
                    let _ = tx.send(UiUpdate::ProviderTested(format!(
                        "provider '{name}' saved (key in Windows Credential Manager)"
                    )));
                    match db.list_providers().await {
                        Ok(rows) => {
                            let mut keys = HashMap::new();
                            for p in &rows {
                                let has = secrets::SecretStore::get(SERVICE, &p.name)
                                    .map(|o| o.is_some())
                                    .unwrap_or(false);
                                keys.insert(p.name.clone(), has);
                            }
                            let _ = tx.send(UiUpdate::Providers(rows));
                            let _ = tx.send(UiUpdate::Keys(keys));
                        }
                        Err(e) => {
                            let _ = tx.send(UiUpdate::Notice(format!("providers: {e}")));
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::ProviderTested(format!("save: {e}")));
                }
            }
        });
    }

    fn delete_provider(&mut self, name: String) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            let _ = secrets::SecretStore::delete(SERVICE, &name);
            match db.delete_provider(&name).await {
                Ok(()) => {
                    let _ = tx.send(UiUpdate::ProviderTested(format!(
                        "provider '{name}' deleted"
                    )));
                    match db.list_providers().await {
                        Ok(rows) => {
                            let _ = tx.send(UiUpdate::Providers(rows));
                        }
                        Err(e) => {
                            let _ = tx.send(UiUpdate::Notice(format!("providers: {e}")));
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(UiUpdate::ProviderTested(format!("delete: {e}")));
                }
            }
        });
    }

    fn test_provider(&mut self, name: String) {
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            let out = match test_provider_inner(&db, &name).await {
                Ok(msg) => msg,
                Err(e) => e,
            };
            let _ = tx.send(UiUpdate::ProviderTested(out));
        });
    }

    /// Self-test: exercises the real database, workspace, policy gate,
    /// filesystem tool, terminal tool and model connection — so a broken
    /// approval pipeline (or anything else) shows up as a named failure
    /// instead of silence. No model tokens spent, nothing modified.
    fn run_diagnostics(&mut self) {
        self.diags_running = true;
        self.diags.clear();
        self.show_diags = true;
        let db = self.db.clone();
        let tx = self.tx.clone();
        let provider_name = self
            .active_id
            .and_then(|id| {
                self.sessions
                    .iter()
                    .find(|s| s.id == id)
                    .and_then(|s| s.provider_name.clone())
            })
            .or_else(|| self.providers.first().map(|p| p.name.clone()));
        self.spawn(async move {
            let mut rows = Vec::new();
            let mut row = |name: &str, r: Result<String, String>| match r {
                Ok(detail) => rows.push(DiagRow {
                    name: name.to_string(),
                    ok: Some(true),
                    detail,
                }),
                Err(detail) => rows.push(DiagRow {
                    name: name.to_string(),
                    ok: Some(false),
                    detail,
                }),
            };

            row(
                "Database",
                match db.list_sessions().await {
                    Ok(s) => Ok(format!("sessions table readable ({} sessions)", s.len())),
                    Err(e) => Err(format!("unreadable: {e}")),
                },
            );

            let dir = default_workspace_dir();
            row(
                "Workspace folder",
                match std::fs::create_dir_all(&dir)
                    .and_then(|_| std::fs::read_dir(&dir).map(|_| ()))
                {
                    Ok(()) => Ok(format!("usable: {}", dir.display())),
                    Err(e) => Err(format!("can't use {}: {e}", dir.display())),
                },
            );

            row("Approval gate", probe_approval_gate().await);
            row("File tools (list)", probe_fs_list(&dir).await);
            row("Terminal tool", probe_process(&dir).await);

            match provider_name {
                Some(name) => {
                    let label = format!("Model connection ({name})");
                    row(&label, test_provider_inner(&db, &name).await);
                }
                None => rows.push(DiagRow {
                    name: "Model connection".to_string(),
                    ok: None,
                    detail: "skipped: add a provider first".to_string(),
                }),
            }

            let _ = tx.send(UiUpdate::Diagnostics(rows));
        });
    }

    /// Open the guided model browser for the currently picked provider.
    fn open_models(&mut self) {
        if self.browser_provider.is_empty() {
            self.browser_provider = if !self.provider_sel.is_empty() {
                self.provider_sel.clone()
            } else {
                self.providers
                    .first()
                    .map(|p| p.name.clone())
                    .unwrap_or_default()
            };
        }
        self.show_models = true;
        self.reload_browser();
    }

    /// (Re)load the model list: live from the server, built-in catalog
    /// as fallback. Never blocks the UI.
    fn reload_browser(&mut self) {
        let name = self.browser_provider.trim().to_string();
        if name.is_empty() {
            return;
        }
        self.browser_loading = true;
        self.browser_models.clear();
        let db = self.db.clone();
        let tx = self.tx.clone();
        self.spawn(async move {
            let err = match load_models_inner(&db, &name).await {
                Ok(models) => {
                    let _ = tx.send(UiUpdate::ModelList {
                        provider: name,
                        models,
                    });
                    return;
                }
                Err(e) => e,
            };
            // Fallback: built-in catalog so picking still works offline.
            let kind = db
                .get_provider(&name)
                .await
                .ok()
                .flatten()
                .map(|r| r.kind)
                .unwrap_or_default();
            let _ = tx.send(UiUpdate::ModelList {
                provider: name.clone(),
                models: known_models(&kind),
            });
            let _ = tx.send(UiUpdate::Notice(format!(
                "Live model list failed ({err}); showing built-in catalog."
            )));
        });
    }

    fn answer_approval(&mut self, id: Uuid, allow: bool) {
        self.approvals.retain(|c| c.approval_id != id);
        let policy = self.policy.clone();
        let tx = self.tx.clone();
        let decision = if allow {
            ApprovalDecision::AllowOnce
        } else {
            ApprovalDecision::Deny
        };
        self.spawn(async move {
            policy.respond(id, decision).await;
            let _ = tx.send(UiUpdate::Notice(
                if allow { "approved" } else { "denied" }.to_string(),
            ));
        });
    }

    // -- update application ------------------------------------------------

    fn drain(&mut self) {
        while let Ok(update) = self.rx.try_recv() {
            match update {
                UiUpdate::Notice(n) => self.notice = Some(n),
                UiUpdate::Sessions(rows) => {
                    self.sessions = rows;
                    self.rebinding = false;
                }
                UiUpdate::SessionCreated(row) => {
                    let id = row.id;
                    self.refresh_sessions();
                    self.open_session(id);
                }
                UiUpdate::SessionMessages { session, messages } => {
                    if self.active_id == Some(session) {
                        self.messages = messages
                            .iter()
                            .filter(|r| r.role == "user" || r.role == "assistant")
                            .map(|r| UiMsg {
                                role: if r.role == "user" {
                                    "user"
                                } else {
                                    "assistant"
                                },
                                content: r.content.clone(),
                                reasoning: r.reasoning.clone().unwrap_or_default(),
                            })
                            .collect();
                    }
                }
                UiUpdate::Providers(rows) => self.providers = rows,
                UiUpdate::Keys(keys) => self.keys = keys,
                UiUpdate::ProviderTested(msg) => self.prov_result = Some(msg),
                UiUpdate::SendFailed(e) => {
                    self.push_activity("err", e.clone());
                    self.notice = Some(e);
                    self.streaming = false;
                }
                UiUpdate::SessionGone(id) => {
                    self.sessions.retain(|s| s.id != id);
                    if self.active_id == Some(id) {
                        self.active_id = None;
                        self.messages.clear();
                        self.stream.clear();
                        self.streaming = false;
                        self.approvals.clear();
                        self.provider_sel.clear();
                        self.model_sel.clear();
                        self.effort_sel = None;
                    }
                }
                UiUpdate::Agent(ev) => self.apply_event(ev),
                UiUpdate::ModelList { provider, models } => {
                    if provider == self.browser_provider {
                        self.browser_models = models;
                        self.browser_loading = false;
                    }
                }
                UiUpdate::SttDone(result) => {
                    self.stt_working = false;
                    match result {
                        Ok(text) => {
                            if text.trim().is_empty() {
                                self.notice = Some(
                                    "Heard nothing — try speaking closer to the mic.".to_string(),
                                );
                            } else {
                                if !self.draft.is_empty()
                                    && !self.draft.ends_with(char::is_whitespace)
                                {
                                    self.draft.push(' ');
                                }
                                self.draft.push_str(text.trim());
                            }
                        }
                        Err(e) => self.notice = Some(e),
                    }
                }
                UiUpdate::Diagnostics(rows) => {
                    self.diags = rows;
                    self.diags_running = false;
                }
            }
        }
    }

    fn apply_event(&mut self, ev: AgentEvent) {
        self.last_agent_event = Some(std::time::Instant::now());
        match ev {
            AgentEvent::TaskCreated { task_id, .. } => {
                self.stream_task = Some(task_id);
                self.stream_reasoning_task = Some(task_id);
                self.streaming = true;
            }
            AgentEvent::ModelDelta { task_id, text, .. } => {
                if self.stream_task == Some(task_id) {
                    self.stream.push_str(&text);
                }
            }
            AgentEvent::ReasoningDelta { task_id, text, .. } => {
                if self.stream_reasoning_task == Some(task_id) {
                    self.stream_reasoning.push_str(&text);
                }
            }
            AgentEvent::MessageCompleted {
                text, reasoning, ..
            } => {
                self.messages.push(UiMsg {
                    role: "assistant",
                    content: text,
                    reasoning,
                });
                self.stream.clear();
                self.stream_reasoning.clear();
            }
            AgentEvent::ToolRequested { tool, args, .. } => {
                let arg_str = serde_json::to_string(&args).unwrap_or_default();
                self.push_activity("tool", format!("{tool}({arg_str})"));
            }
            AgentEvent::ToolOutput { tool, output, .. } => {
                let mut short: String = output.chars().take(300).collect();
                if output.len() > short.len() {
                    short.push('…');
                }
                self.push_activity("output", format!("{tool}: {short}"));
            }
            AgentEvent::ApprovalRequested {
                approval_id,
                tool,
                args,
                risk,
                reason,
                ..
            } => self.approvals.push(ApprovalCard {
                approval_id,
                tool,
                args,
                risk: risk.label().to_string(),
                reason,
            }),
            AgentEvent::ApprovalResolved { approval_id, .. } => {
                self.approvals.retain(|c| c.approval_id != approval_id);
            }
            AgentEvent::TaskCompleted { summary, .. } => {
                self.push_activity(
                    "ok",
                    format!(
                        "task done · {} turns · {}+{} tokens",
                        summary.turns, summary.input_tokens, summary.output_tokens
                    ),
                );
                self.last_summary = Some(summary);
                self.streaming = false;
                self.stream_reasoning.clear();
            }
            AgentEvent::TaskFailed { error, .. } => {
                self.push_activity("err", error.clone());
                // Loud failure: non-technical users never open the Activity
                // panel, so the error must surface in the conversation itself.
                self.messages.push(UiMsg {
                    role: "assistant",
                    content: format!("I ran into a problem and stopped:\n\n{error}"),
                    reasoning: String::new(),
                });
                self.notice = Some(error);
                self.stream.clear();
                self.stream_reasoning.clear();
                self.streaming = false;
            }
            _ => {}
        }
    }
}

/// One-line catalog summary: context window + prices + cutoff.
fn catalog_line(m: &ModelInfo) -> String {
    let mut parts = Vec::new();
    if let Some(ctx) = m.context_window {
        if ctx >= 1_000_000 {
            parts.push(format!("{:.2}M ctx", ctx as f64 / 1_000_000.0));
        } else {
            parts.push(format!("{}K ctx", ctx / 1_000));
        }
    }
    if let (Some(i), Some(o)) = (m.input_price_per_mtok, m.output_price_per_mtok) {
        parts.push(format!("${i}/${o} per MTok"));
    }
    if let Some(cutoff) = &m.knowledge_cutoff {
        parts.push(format!("cutoff {cutoff}"));
    }
    if let Some(notes) = &m.notes {
        parts.push(notes.clone());
    }
    if parts.is_empty() {
        "no metadata".to_string()
    } else {
        parts.join(" · ")
    }
}

/// Human-friendly label for a bound model: the catalog display name when
/// the id is known, otherwise the raw id.
fn friendly_model(providers: &[ProviderRow], provider: &str, model: &str) -> String {
    let kind = providers
        .iter()
        .find(|p| p.name == provider)
        .map(|p| p.kind.as_str())
        .unwrap_or("");
    known_models(kind)
        .into_iter()
        .find(|m| m.id == model)
        .map(|m| m.display_name)
        .unwrap_or_else(|| model.to_string())
}

fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Provider construction (BYOK; keys never leave Rust / the OS store)
// ---------------------------------------------------------------------------

fn stored_key(name: &str) -> Result<Option<SecretString>, String> {
    secrets::SecretStore::get(SERVICE, name).map_err(|e| format!("key store: {e}"))
}

fn require_key(name: &str) -> Result<SecretString, String> {
    match stored_key(name)? {
        Some(k) => Ok(k),
        None => Err(format!(
            "no key stored for '{name}' — add it in Providers & keys"
        )),
    }
}

fn build_provider(
    kind: &str,
    base_url: Option<String>,
    key: Option<SecretString>,
    default_model: Option<String>,
) -> Result<Arc<dyn ModelProvider>, String> {
    match kind {
        "openai" => {
            let key = key.ok_or_else(|| "provider key missing".to_string())?;
            Ok(Arc::new(provider_openai::OpenAiProvider::new(
                key,
                base_url,
                default_model,
            )))
        }
        "anthropic" => {
            let key = key.ok_or_else(|| "provider key missing".to_string())?;
            Ok(Arc::new(provider_anthropic::AnthropicProvider::new(
                key,
                base_url,
                default_model,
            )))
        }
        "xai" => {
            let key = key.ok_or_else(|| "provider key missing".to_string())?;
            Ok(Arc::new(provider_xai::XaiProvider::new(
                key,
                base_url,
                default_model,
            )))
        }
        "mistral" => {
            let key = key.ok_or_else(|| "provider key missing".to_string())?;
            Ok(Arc::new(provider_compat::CompatProvider::mistral(
                key,
                base_url,
                default_model,
            )))
        }
        "gemini" => {
            let key = key.ok_or_else(|| "provider key missing".to_string())?;
            Ok(Arc::new(provider_compat::CompatProvider::gemini(
                key,
                base_url,
                default_model,
            )))
        }
        "ollama" => Ok(Arc::new(provider_compat::CompatProvider::ollama(
            base_url,
            default_model,
        ))),
        "local" => Ok(Arc::new(provider_compat::CompatProvider::local(
            key,
            base_url,
            default_model,
        ))),
        other => Err(format!("unsupported provider kind: {other}")),
    }
}

fn known_models(kind: &str) -> Vec<ModelInfo> {
    match kind {
        "openai" => provider_openai::models::known_models(),
        "anthropic" => provider_anthropic::models::known_models(),
        "xai" => provider_xai::models::known_models(),
        "mistral" => provider_compat::models::known_models_mistral(),
        "gemini" => provider_compat::models::known_models_gemini(),
        "ollama" => provider_compat::models::known_models_ollama(),
        "local" => provider_compat::models::known_models_local(),
        _ => Vec::new(),
    }
}

async fn provider_for(
    db: &Db,
    name: &str,
) -> Result<(Arc<dyn ModelProvider>, ProviderRow), String> {
    let row = db
        .get_provider(name)
        .await
        .map_err(|e| format!("{e}"))?
        .ok_or_else(|| format!("provider '{name}' not found"))?;
    // Local kinds work without a key; cloud kinds require one.
    let key = if key_required(&row.kind) {
        Some(require_key(name)?)
    } else {
        stored_key(name)?
    };
    let provider = build_provider(
        &row.kind,
        row.base_url.clone(),
        key,
        row.default_model.clone(),
    )?;
    Ok((provider, row))
}

async fn load_models_inner(db: &Db, name: &str) -> Result<Vec<ModelInfo>, String> {
    let (provider, _) = provider_for(db, name).await?;
    provider.list_models().await.map_err(|e| format!("{e}"))
}

/// Approval-gate probe: runs a fake medium-risk write through a fresh
/// policy engine and completes the human round-trip. Proves the exact
/// machinery behind approval cards works (raise → answer → resolve).
async fn probe_approval_gate() -> Result<String, String> {
    let policy = PolicyEngine::new(RiskClass::Low);
    let def = tool_core::ToolDefinition::new(
        "diag.probe",
        "diagnostic probe (never executed)",
        serde_json::json!({"type": "object"}),
        RiskClass::Medium,
        vec![SideEffect::WritesFiles],
    );
    match policy
        .authorize(&def, &serde_json::Value::Null, &ToolContext::default())
        .await
    {
        Decision::PendingApproval { id, receiver, .. } => {
            policy.respond(id, ApprovalDecision::AllowOnce).await;
            match receiver.await {
                Ok(ApprovalDecision::AllowOnce) => {
                    Ok("raise → answer → resolve round-trip works".to_string())
                }
                Ok(other) => Err(format!("gate answered wrong: {other:?}")),
                Err(_) => Err("approval answer got lost".to_string()),
            }
        }
        Decision::Allow => Err("expected an approval gate, got auto-allow".to_string()),
        Decision::Deny { reason } => Err(format!("unexpected deny: {reason}")),
    }
}

/// Filesystem probe: runs the real `fs.list` tool against the workspace.
async fn probe_fs_list(dir: &std::path::Path) -> Result<String, String> {
    let tools = tool_filesystem::filesystem_tools(dir.to_path_buf());
    let list = tools
        .into_iter()
        .find(|t| t.definition().name == "fs.list")
        .ok_or_else(|| "fs.list tool missing".to_string())?;
    let ctx = ToolContext {
        workspace: Some(WorkspaceConfig::from_dir("diag", dir.to_path_buf())),
        cwd: Some(dir.to_path_buf()),
    };
    let out = list
        .execute(serde_json::json!({"path": "."}), &ctx)
        .await
        .map_err(|e| format!("{e:#}"))?;
    Ok(format!("fs.list works ({} chars listed)", out.output.len()))
}

/// Terminal probe: runs one harmless echo through the real `process.run`.
async fn probe_process(dir: &std::path::Path) -> Result<String, String> {
    let tools = tool_process::process_tools();
    let run = tools
        .into_iter()
        .find(|t| t.definition().name == "process.run")
        .ok_or_else(|| "process.run tool missing".to_string())?;
    let ctx = ToolContext {
        workspace: Some(WorkspaceConfig::from_dir("diag", dir.to_path_buf())),
        cwd: Some(dir.to_path_buf()),
    };
    let out = tokio::time::timeout(
        std::time::Duration::from_secs(25),
        run.execute(
            serde_json::json!({"shell": "powershell", "command": "Write-Output diag-ok", "timeout_ms": 15000}),
            &ctx,
        ),
    )
    .await
    .map_err(|_| "terminal call timed out".to_string())
    .and_then(|r| r.map_err(|e| format!("{e:#}")))?;
    if out.output.contains("diag-ok") && out.output.contains("exit: 0") {
        Ok("PowerShell round-trip works".to_string())
    } else {
        Err(format!(
            "unexpected output: {}",
            out.output.chars().take(200).collect::<String>()
        ))
    }
}

async fn test_provider_inner(db: &Db, name: &str) -> Result<String, String> {
    let (provider, row) = provider_for(db, name).await?;
    match provider.list_models().await {
        Ok(models) => {
            if models.is_empty() {
                Ok(format!(
                    "Connected to {} — no models reported",
                    provider.kind().label()
                ))
            } else {
                Ok(format!(
                    "Connected to {} — {} models available (e.g. {})",
                    provider.kind().label(),
                    models.len(),
                    models[0].id
                ))
            }
        }
        Err(list_err) => {
            // Some compat servers (or proxies) skip /models: fall back to a
            // tiny chat ping so a working endpoint still tests green.
            let model = row
                .default_model
                .clone()
                .or_else(|| known_models(&row.kind).first().map(|m| m.id.clone()))
                .unwrap_or_else(|| "default".to_string());
            let req = provider_api::AgentRequest {
                model,
                messages: vec![Message::user("Reply with exactly: ok")],
                tools: Vec::new(),
                system: None,
                temperature: None,
                max_tokens: Some(16),
                reasoning_effort: None,
            };
            let mut stream = provider
                .stream_response(&req)
                .await
                .map_err(|e| format!("{list_err}; ping: {e}"))?;
            let mut text = String::new();
            let timeout = tokio::time::sleep(std::time::Duration::from_secs(60));
            tokio::pin!(timeout);
            loop {
                tokio::select! {
                    next = stream.next() => {
                        match next {
                            Some(provider_api::ProviderEvent::TextDelta { text: t }) => text.push_str(&t),
                            Some(provider_api::ProviderEvent::Done) | None => break,
                            Some(provider_api::ProviderEvent::Error { message }) => {
                                return Err(format!("{list_err}; ping: {message}"));
                            }
                            _ => {}
                        }
                    }
                    _ = &mut timeout => {
                        return Err(format!("{list_err}; ping timed out"));
                    }
                }
            }
            Ok(format!(
                "Connected to {} — reachable (model list unavailable: {list_err}; ping: {text})",
                provider.kind().label()
            ))
        }
    }
}

async fn send_message_inner(
    db: &Db,
    agent: &Arc<Agent>,
    session_id: Uuid,
    text: String,
) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("message is empty".into());
    }
    let session = db
        .get_session(session_id)
        .await
        .map_err(|e| format!("{e}"))?
        .ok_or_else(|| "session not found".to_string())?;

    let provider_name = session
        .provider_name
        .clone()
        .ok_or_else(|| "no provider set for this session — pick one in the header".to_string())?;
    let row = db
        .get_provider(&provider_name)
        .await
        .map_err(|e| format!("{e}"))?
        .ok_or_else(|| format!("provider '{provider_name}' not found"))?;
    let key = if key_required(&row.kind) {
        Some(require_key(&provider_name)?)
    } else {
        stored_key(&provider_name)?
    };
    let provider = build_provider(
        &row.kind,
        row.base_url.clone(),
        key,
        row.default_model.clone(),
    )?;
    let model = session
        .model
        .clone()
        .or_else(|| row.default_model.clone())
        .ok_or_else(|| "no model set for this session".to_string())?;

    let rows = db
        .list_messages(session_id)
        .await
        .map_err(|e| format!("{e}"))?;
    let history: Vec<Message> = rows.iter().filter_map(to_internal_message).collect();

    // Every session gets a confined root: the bound folder, or the
    // ~/Super-AI fallback for older sessions. File + terminal tools are
    // always attached, so listing/reading/writing and PowerShell just work.
    let workspace_path = session.workspace.clone().or_else(|| {
        let dir = default_workspace_dir();
        let _ = std::fs::create_dir_all(&dir);
        Some(dir.to_string_lossy().to_string())
    });
    let workspace = workspace_path
        .as_deref()
        .map(|w| WorkspaceConfig::from_dir(session.title.clone(), PathBuf::from(w)));
    let mut tools: Vec<Arc<dyn tool_core::Tool>> = Vec::new();
    if let Some(root) = workspace.as_ref().and_then(|ws| ws.root()).cloned() {
        tools.extend(tool_filesystem::filesystem_tools(root));
        tools.extend(tool_process::process_tools());
    }
    let registry = ToolRegistry::new(tools);

    db.insert_message(&MessageRow {
        id: Uuid::new_v4(),
        session_id,
        role: "user".into(),
        content: text.clone(),
        tool_calls: None,
        tool_call_id: None,
        reasoning: None,
        created_at: now_ms(),
    })
    .await
    .map_err(|e| format!("{e}"))?;
    db.touch_session(session_id)
        .await
        .map_err(|e| format!("{e}"))?;

    let mut system = SYSTEM_PROMPT.to_string();
    // The operating mode changes what the model may even attempt: in Plan
    // mode the policy refuses writes/commands, so the prompt must steer
    // toward investigating and explaining instead.
    if agent.policy.mode().await == AgentMode::Build {
        system.push_str(
            "\n\nCurrent mode: BUILD — use any tool you need. File writes and terminal \
             commands need human approval and may be denied; if so, report it and adapt.",
        );
    } else {
        system.push_str(
            "\n\nCurrent mode: PLAN (read-only) — ONLY use fs.list and fs.read to investigate, \
             then explain your plan step by step. Do NOT call fs.write or process.run: they \
             are blocked in this mode. End by asking the user to switch to Build mode (Tab \
             key) if they want you to implement it.",
        );
    }
    if let Some(root) = workspace.as_ref().and_then(|ws| ws.root()) {
        system.push_str(&format!(
            "\n\nYour workspace folder is: {}\nList, read and write files there, and run terminal commands from there.",
            root.display()
        ));
    }
    let reasoning_effort = session
        .reasoning_effort
        .as_deref()
        .and_then(ReasoningEffort::parse);

    let request = TaskRequest {
        session_id,
        provider,
        model,
        system_prompt: Some(system),
        history,
        user_message: text,
        workspace,
        tools: registry,
        max_turns: 32,
        reasoning_effort,
    };

    // The task runs under the agent's supervision (panic reporting,
    // abort registry); its outcome arrives via the event bus.
    agent.spawn(request);
    Ok(())
}

fn to_internal_message(row: &MessageRow) -> Option<Message> {
    let role = match row.role.as_str() {
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "system" => MessageRole::System,
        "tool" => MessageRole::Tool,
        _ => return None,
    };
    let tool_calls = row
        .tool_calls
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();
    Some(Message {
        role,
        content: row.content.clone(),
        tool_calls,
        tool_call_id: row.tool_call_id.clone(),
    })
}

// ---------------------------------------------------------------------------
// Event persistence (flight recorder), mirrored from the old shell
// ---------------------------------------------------------------------------

fn event_kind(ev: &AgentEvent) -> &'static str {
    match ev {
        AgentEvent::TaskCreated { .. } => "task_created",
        AgentEvent::ReasoningStarted { .. } => "reasoning_started",
        AgentEvent::ModelDelta { .. } => "model_delta",
        AgentEvent::ReasoningDelta { .. } => "reasoning_delta",
        AgentEvent::ToolRequested { .. } => "tool_requested",
        AgentEvent::ToolStarted { .. } => "tool_started",
        AgentEvent::ToolOutput { .. } => "tool_output",
        AgentEvent::ApprovalRequested { .. } => "approval_requested",
        AgentEvent::ApprovalResolved { .. } => "approval_resolved",
        AgentEvent::MessageCompleted { .. } => "message_completed",
        AgentEvent::TaskFailed { .. } => "task_failed",
        AgentEvent::TaskCompleted { .. } => "task_completed",
    }
}

fn event_ids(ev: &AgentEvent) -> (Option<Uuid>, Option<Uuid>) {
    match ev {
        AgentEvent::TaskCreated {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ReasoningStarted {
            task_id,
            session_id,
        }
        | AgentEvent::ModelDelta {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ReasoningDelta {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ToolRequested {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ToolStarted {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ToolOutput {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ApprovalRequested {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::ApprovalResolved {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::MessageCompleted {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::TaskFailed {
            task_id,
            session_id,
            ..
        }
        | AgentEvent::TaskCompleted {
            task_id,
            session_id,
            ..
        } => (Some(*task_id), Some(*session_id)),
    }
}

async fn persist_event(db: &Db, ev: &AgentEvent) {
    // Token deltas stream to the UI live; writing every one to SQLite would
    // drown the flight recorder (and stall the event bus). Everything else
    // is persisted.
    if matches!(
        ev,
        AgentEvent::ModelDelta { .. } | AgentEvent::ReasoningDelta { .. }
    ) {
        return;
    }
    let payload = match serde_json::to_value(ev) {
        Ok(p) => p,
        Err(_) => return,
    };
    let (task_id, session_id) = event_ids(ev);
    let _ = db
        .insert_event(task_id, session_id, event_kind(ev), &payload)
        .await;
    // Mirror the conversation into `messages` so the next turn reloads a
    // complete, correctly paired history (assistant tool_calls + tool
    // results). Without this, providers reject follow-up requests.
    match ev {
        AgentEvent::MessageCompleted {
            session_id,
            text,
            reasoning,
            tool_calls,
            ..
        } => {
            let tool_calls = if tool_calls.is_empty() {
                None
            } else {
                serde_json::to_string(tool_calls).ok()
            };
            let reasoning = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning.clone())
            };
            let _ = db
                .insert_message(&MessageRow {
                    id: Uuid::new_v4(),
                    session_id: *session_id,
                    role: "assistant".into(),
                    content: text.clone(),
                    tool_calls,
                    tool_call_id: None,
                    reasoning,
                    created_at: now_ms(),
                })
                .await;
        }
        AgentEvent::ToolOutput {
            session_id,
            tool_call_id,
            output,
            ..
        } => {
            let _ = db
                .insert_message(&MessageRow {
                    id: Uuid::new_v4(),
                    session_id: *session_id,
                    role: "tool".into(),
                    content: output.clone(),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.clone()),
                    reasoning: None,
                    created_at: now_ms(),
                })
                .await;
        }
        _ => {}
    }
    if let Some(sid) = session_id {
        let _ = db.touch_session(sid).await;
    }
}

// ---------------------------------------------------------------------------
// egui rendering
// ---------------------------------------------------------------------------

impl eframe::App for SuperAiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain();
        // Global stop gesture: ESC twice kills whatever the agent is doing.
        // (Consumed here so typing in the composer can trigger it too;
        // Shift+ESC and other combos keep their normal behavior.)
        let esc = ui
            .input_mut(|i| i.count_and_consume_key(egui::Modifiers::NONE, egui::Key::Escape) != 0);
        if esc {
            self.handle_esc();
        }
        // Ctrl+Tab toggles Plan/Build from anywhere (Tab alone also works
        // while typing in the composer).
        let mode_key =
            ui.input_mut(|i| i.count_and_consume_key(egui::Modifiers::CTRL, egui::Key::Tab) != 0);
        if mode_key {
            self.toggle_mode();
        }
        let ctx = ui.ctx().clone();

        let active = self
            .active_id
            .and_then(|id| self.sessions.iter().find(|s| s.id == id).cloned());

        // -- top bar: brand + session binding -------------------------------
        egui::Panel::top("brand").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(li(Icon::Bot).size(20.0).strong());
                ui.heading("Super-AI");
                ui.label(egui::RichText::new("v0.2.0").weak().small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if active.is_some() {
                        if let Some(s) = &self.last_summary {
                            ui.label(
                                egui::RichText::new(format!(
                                    "{} turns · {}+{} tok",
                                    s.turns, s.input_tokens, s.output_tokens
                                ))
                                .small()
                                .weak(),
                            );
                        }
                        let model_label = if self.model_sel.trim().is_empty() {
                            "Choose model…".to_string()
                        } else {
                            friendly_model(&self.providers, &self.provider_sel, &self.model_sel)
                        };
                        if ui
                            .button(model_label)
                            .on_hover_text("Pick a model — switches instantly, history kept")
                            .clicked()
                        {
                            self.open_models();
                        }
                        egui::ComboBox::from_id_salt("bind_provider")
                            .selected_text(if self.provider_sel.is_empty() {
                                "provider…"
                            } else {
                                &self.provider_sel
                            })
                            .width(150.0)
                            .show_ui(ui, |ui| {
                                if self.providers.is_empty() {
                                    ui.label(
                                        egui::RichText::new(
                                            "No providers yet — add one in Providers & keys.",
                                        )
                                        .small()
                                        .weak(),
                                    );
                                }
                                for p in &self.providers {
                                    ui.selectable_value(
                                        &mut self.provider_sel,
                                        p.name.clone(),
                                        format!("{} ({})", p.name, kind_label(&p.kind)),
                                    );
                                }
                            });
                        if self.providers.is_empty() {
                            ui.label(egui::RichText::new("no providers").small().weak());
                            if ui.button("Add provider").clicked() {
                                self.show_providers = true;
                            }
                        }
                        // Seamless switching: a complete selection that
                        // differs from the stored binding rebinds instantly —
                        // no Apply step, history kept, takes effect on the
                        // next message (an in-flight answer keeps its model).
                        if !self.rebinding {
                            let want_p = self.provider_sel.trim().to_string();
                            let want_m = self.model_sel.trim().to_string();
                            if let Some(s) = &active {
                                let bound_p = s.provider_name.clone().unwrap_or_default();
                                let bound_m = s.model.clone().unwrap_or_default();
                                if !want_p.is_empty()
                                    && !want_m.is_empty()
                                    && (want_p != bound_p || want_m != bound_m)
                                {
                                    self.rebinding = true;
                                    self.apply_binding();
                                }
                            }
                        }
                    } else {
                        ui.label("← create a session to start");
                    }
                });
            });
            if let Some(s) = &active {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(&s.title).strong());
                    if let Some(ws) = &s.workspace {
                        ui.label(egui::RichText::new(ws).small().monospace());
                    }
                    if let Some(p) = &s.provider_name {
                        ui.label(
                            egui::RichText::new(format!(
                                "{} · {}",
                                p,
                                s.model.as_deref().unwrap_or("no model")
                            ))
                            .small()
                            .weak(),
                        );
                    }
                });
            }
            ui.add_space(4.0);
        });

        // -- left: sessions + providers --------------------------------------
        egui::Panel::left("sidebar")
            .resizable(true)
            .default_size(240.0)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(li(Icon::MessageSquare).strong());
                    ui.heading("Sessions");
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.new_workspace)
                            .hint_text("Folder for files (blank = Super-AI folder)")
                            .desired_width(f32::INFINITY),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label(li(Icon::Plus));
                    if ui.button("New session").clicked() {
                        self.create_session();
                    }
                });
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.sessions.is_empty() {
                            ui.label(egui::RichText::new("No sessions yet").weak());
                        }
                        let mut open: Option<Uuid> = None;
                        let mut gone: Option<Uuid> = None;
                        for s in &self.sessions {
                            let selected = self.active_id == Some(s.id);
                            let title = format!(
                                "{}\n{} · {}",
                                s.title,
                                s.provider_name.as_deref().unwrap_or("no provider"),
                                s.model.as_deref().unwrap_or("no model")
                            );
                            ui.horizontal(|ui| {
                                let resp = ui.add_sized(
                                    egui::vec2(ui.available_width() - 30.0, 0.0),
                                    egui::Button::selectable(selected, title),
                                );
                                if resp.clicked() {
                                    open = Some(s.id);
                                }
                                if ui
                                    .button(li(Icon::X))
                                    .on_hover_text("Delete session")
                                    .clicked()
                                {
                                    gone = Some(s.id);
                                }
                            });
                        }
                        if let Some(id) = open {
                            self.open_session(id);
                        }
                        if let Some(id) = gone {
                            self.delete_session(id);
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(li(Icon::SlidersHorizontal));
                    ui.label("Autonomy");
                    egui::ComboBox::from_id_salt("autonomy")
                        .selected_text(self.autonomy.label())
                        .show_ui(ui, |ui| {
                            for r in [RiskClass::Low, RiskClass::Medium, RiskClass::High] {
                                if ui
                                    .selectable_value(&mut self.autonomy, r, r.label())
                                    .clicked()
                                {
                                    self.set_autonomy(r);
                                }
                            }
                        });
                });
                ui.label(
                    egui::RichText::new("Auto-approves at or below this risk.")
                        .small()
                        .weak(),
                );
                ui.horizontal(|ui| {
                    ui.label(li(Icon::Brain));
                    ui.label("Reasoning");
                    ui.add_enabled_ui(self.active_id.is_some(), |ui| {
                        let before = self.effort_sel;
                        egui::ComboBox::from_id_salt("effort")
                            .selected_text(self.effort_sel.map(|e| e.label()).unwrap_or("Default"))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.effort_sel, None, "Default");
                                for e in [
                                    ReasoningEffort::Off,
                                    ReasoningEffort::Low,
                                    ReasoningEffort::Medium,
                                    ReasoningEffort::High,
                                    ReasoningEffort::Max,
                                ] {
                                    ui.selectable_value(&mut self.effort_sel, Some(e), e.label());
                                }
                            });
                        if self.effort_sel != before {
                            let effort = self.effort_sel;
                            if let Some(id) = self.active_id {
                                let db = self.db.clone();
                                let s = effort.map(|e| e.as_str().to_string());
                                self.spawn(async move {
                                    let _ = db.set_session_effort(id, s.as_deref()).await;
                                });
                            }
                        }
                    });
                });
                ui.label(
                    egui::RichText::new("How hard this session's model thinks.")
                        .small()
                        .weak(),
                );
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    ui.add_space(4.0);
                    ui.label(egui::RichText::new("v0.2.0 · BYOK").weak().small());
                    ui.horizontal(|ui| {
                        ui.label(li(Icon::Stethoscope));
                        if ui
                            .button("Diagnostics")
                            .on_hover_text("Self-test: database, tools, approval gate, model")
                            .clicked()
                        {
                            self.run_diagnostics();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(li(Icon::Settings));
                        if ui.button("Providers & keys").clicked() {
                            self.show_providers = true;
                        }
                    });
                });
            });

        // -- bottom: mode strip + composer --------------------------------------
        egui::Panel::bottom("composer").show(ui, |ui| {
            ui.add_space(4.0);
            if let Some(n) = self.notice.clone() {
                ui.label(egui::RichText::new(n).small().weak());
            }
            // Operating mode: blue Build acts (with approvals), yellow Plan
            // only reads. Clickable pills + Tab toggle in the composer.
            let mut toggle = false;
            ui.horizontal(|ui| {
                let build_on = self.mode_mirror == AgentMode::Build;
                if ui
                    .add(egui::Button::selectable(
                        build_on,
                        egui::RichText::new("Build")
                            .strong()
                            .color(egui::Color32::from_rgb(96, 165, 250)),
                    ))
                    .on_hover_text("Build mode: the agent can act (approvals still apply)")
                    .clicked()
                    && !build_on
                {
                    toggle = true;
                }
                if ui
                    .add(egui::Button::selectable(
                        !build_on,
                        egui::RichText::new("Plan")
                            .strong()
                            .color(egui::Color32::from_rgb(250, 204, 21)),
                    ))
                    .on_hover_text("Plan mode: read-only, the agent plans but changes nothing")
                    .clicked()
                    && build_on
                {
                    toggle = true;
                }
                ui.label(
                    egui::RichText::new(if build_on {
                        "acts with approval prompts"
                    } else {
                        "read-only — Tab to switch"
                    })
                    .small()
                    .weak(),
                );
            });
            ui.horizontal(|ui| {
                let hint = if self.streaming {
                    "Agent is working…"
                } else if active
                    .as_ref()
                    .and_then(|s| s.provider_name.as_ref())
                    .is_some()
                {
                    "Ask the agent something… (Enter sends · Tab / Ctrl+Tab switches mode · ESC×2 stops)"
                } else {
                    "Bind a provider and model above first…"
                };
                let resp = ui.add(
                    egui::TextEdit::multiline(&mut self.draft)
                        .id(egui::Id::new("composer"))
                        .desired_rows(2)
                        .hint_text(hint)
                        .desired_width(f32::INFINITY),
                );
                let enter = resp.has_focus()
                    && ui.input_mut(|i| {
                        i.count_and_consume_key(egui::Modifiers::NONE, egui::Key::Enter) != 0
                    });
                // Tab toggles Plan/Build while typing (focus navigation
                // elsewhere is untouched).
                let tab = resp.has_focus()
                    && ui.input_mut(|i| {
                        i.count_and_consume_key(egui::Modifiers::NONE, egui::Key::Tab) != 0
                    });
                if tab {
                    toggle = true;
                }
                // Microphone right of the message bar: toggle to dictate.
                let rec_on = self.stt.is_some();
                if ui
                    .add_enabled(
                        !self.stt_working,
                        egui::Button::selectable(
                            rec_on,
                            li(if rec_on { Icon::Square } else { Icon::Mic }).strong(),
                        ),
                    )
                    .on_hover_text(if rec_on {
                        "Stop and transcribe"
                    } else {
                        "Dictate with microphone"
                    })
                    .clicked()
                {
                    if self.stt.is_some() {
                        self.stop_and_transcribe();
                    } else {
                        self.start_mic();
                    }
                }
                if let Some(rec) = &self.stt {
                    let secs = rec.elapsed().as_secs();
                    ui.label(
                        egui::RichText::new(format!("● {:02}:{:02}", secs / 60, secs % 60))
                            .small()
                            .strong()
                            .color(egui::Color32::LIGHT_RED),
                    );
                }
                // Send stays clickable so it always answers: empty draft
                // and busy agent explain themselves instead of dead clicks.
                let can_send = self.active_id.is_some();
                if ui
                    .add_enabled(can_send, egui::Button::new("Send"))
                    .on_hover_text("Send (Enter)")
                    .clicked()
                    && can_send
                {
                    if self.draft.trim().is_empty() {
                        self.notice = Some("Type a message first.".to_string());
                    } else if self.streaming {
                        self.notice = Some(
                            "The agent is still working — ESC×2 stops it, or wait for it to finish."
                                .to_string(),
                        );
                    } else {
                        self.queue_send();
                    }
                }
                if enter && can_send && !self.streaming && !self.draft.trim().is_empty() {
                    self.queue_send();
                }
            });
            if toggle {
                self.toggle_mode();
            }
            // Recording auto-stops at the cap so a forgotten mic can't run on.
            if let Some(rec) = &self.stt
                && rec.elapsed() > std::time::Duration::from_secs(stt::MAX_RECORD_SECS)
            {
                self.stop_and_transcribe();
            }
            // Focus the composer after opening a session.
            if self.focus_composer {
                self.focus_composer = false;
                ui.memory_mut(|m| m.request_focus(egui::Id::new("composer")));
            }
            ui.add_space(4.0);
        });

        // -- center: conversation + approvals + activity ----------------------
        egui::CentralPanel::default().show(ui, |ui| {
            // Unmissable banner while anything awaits approval.
            if !self.approvals.is_empty() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(
                            li(Icon::TriangleAlert)
                                .strong()
                                .color(egui::Color32::from_rgb(250, 204, 21)),
                        );
                        ui.label(
                            egui::RichText::new(format!(
                                "{} action{} need{} your approval to continue — see below",
                                self.approvals.len(),
                                if self.approvals.len() == 1 { "" } else { "s" },
                                if self.approvals.len() == 1 { "s" } else { "" },
                            ))
                            .strong(),
                        );
                    });
                });
            }
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if active.is_none() {
                        // First-run guide for non-technical users.
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            ui.label(li(Icon::Bot).size(22.0).strong());
                            ui.heading("Welcome to Super-AI");
                        });
                        ui.label(
                            egui::RichText::new(
                                "Your AI assistant for files and terminal tasks. Three steps:",
                            )
                            .weak(),
                        );
                        ui.add_space(8.0);
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(li(Icon::KeyRound).strong());
                                ui.label(egui::RichText::new("1. Add a provider").strong());
                            });
                            ui.label(
                                "Pick ChatGPT, Claude, Grok, Mistral, Gemini — or run free \
                                 local models with Ollama. Keys stay in Windows Credential Manager.",
                            );
                            if ui.button("Open Providers & keys").clicked() {
                                self.show_providers = true;
                            }
                        });
                        ui.add_space(6.0);
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(li(Icon::MessageSquare).strong());
                                ui.label(egui::RichText::new("2. Create a session").strong());
                            });
                            ui.label(
                                "Type a folder path in the sidebar, or leave it blank to use \
                                 your Super-AI folder, then press New session. The AI can \
                                 list, read and write files there and run terminal commands.",
                            );
                            if ui.button("New session").clicked() {
                                self.create_session();
                            }
                        });
                        ui.add_space(6.0);
                        egui::Frame::group(ui.style()).show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.label(li(Icon::Bot).strong());
                                ui.label(
                                    egui::RichText::new("3. Choose a model and chat").strong(),
                                );
                            });
                            ui.label(
                                "Use the provider + model controls at the top (changeable any \
                                 time, even mid-conversation), then ask for something.",
                            );
                        });
                    } else if self.messages.is_empty() && self.stream.is_empty() {
                        ui.add_space(20.0);
                        ui.label(
                            egui::RichText::new(
                                "Pick a provider and press \"Choose model…\" above if you \
                                 haven't yet — then ask the agent to do something.\n\
                                 It can read/write files and run terminal commands (approvals required).",
                            )
                            .weak(),
                        );
                    }
                    for (idx, m) in self.messages.iter().enumerate() {
                        ui.add_space(6.0);
                        let (ic, head, color) = if m.role == "user" {
                            (Icon::User, "you", egui::Color32::LIGHT_BLUE)
                        } else {
                            (Icon::Bot, "assistant", egui::Color32::LIGHT_GREEN)
                        };
                        ui.push_id(format!("msg-{idx}"), |ui| {
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(li(ic).color(color));
                                    ui.label(
                                        egui::RichText::new(head).small().strong().color(color),
                                    );
                                });
                                if !m.reasoning.is_empty() {
                                    egui::CollapsingHeader::new("Thinking")
                                        .default_open(false)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(&m.reasoning)
                                                    .small()
                                                    .weak(),
                                            );
                                        });
                                }
                                md::show(ui, &format!("msg-{idx}"), &m.content);
                            });
                        });
                    }
                    if !self.stream.is_empty() || !self.stream_reasoning.is_empty() {
                        // Snapshot counts first: the header below borrows them
                        // while the body streams live token-by-token.
                        let live_chars = self.stream.chars().count()
                            + self.stream_reasoning.chars().count();
                        ui.add_space(6.0);
                        ui.push_id("streaming", |ui| {
                            egui::Frame::group(ui.style()).show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(li(Icon::Bot).color(egui::Color32::LIGHT_GREEN));
                                    ui.label(
                                        egui::RichText::new(format!(
                                            "assistant · LIVE · {live_chars} chars"
                                        ))
                                        .small()
                                        .strong()
                                        .color(egui::Color32::LIGHT_GREEN),
                                    );
                                });
                                if !self.stream_reasoning.is_empty() {
                                    egui::CollapsingHeader::new("Thinking")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            ui.label(
                                                egui::RichText::new(&self.stream_reasoning)
                                                    .small()
                                                    .weak(),
                                            );
                                        });
                                }
                                if !self.stream.is_empty() {
                                    md::show(ui, "streaming", &format!("{}▍", self.stream));
                                }
                            });
                        });
                    }
                });

            if !self.approvals.is_empty() {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(li(Icon::ShieldAlert).strong());
                    ui.heading("Approvals");
                });
                let mut answer: Option<(Uuid, bool)> = None;
                for card in &self.approvals {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let color = match card.risk.as_str() {
                                "HIGH" | "CRITICAL" => egui::Color32::LIGHT_RED,
                                "MEDIUM" => egui::Color32::GOLD,
                                _ => egui::Color32::GRAY,
                            };
                            ui.label(li(Icon::TriangleAlert).color(color));
                            ui.label(egui::RichText::new(&card.risk).strong().color(color));
                            ui.label(egui::RichText::new(&card.tool).strong().monospace());
                        });
                        let args = serde_json::to_string_pretty(&card.args)
                            .unwrap_or_else(|_| "{}".into());
                        ui.monospace(args);
                        ui.label(egui::RichText::new(&card.reason).small().weak());
                        ui.horizontal(|ui| {
                            if ui.button("Allow once").clicked() {
                                answer = Some((card.approval_id, true));
                            }
                            if ui.button("Deny").clicked() {
                                answer = Some((card.approval_id, false));
                            }
                        });
                    });
                }
                if let Some((id, allow)) = answer {
                    self.answer_approval(id, allow);
                }
            }

            if !self.activity.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new("Activity")
                    .default_open(true)
                    .show(ui, |ui| {
                        for a in self.activity.iter().rev().take(20) {
                            let (ic, color) = match a.kind {
                                "err" => (Icon::X, egui::Color32::LIGHT_RED),
                                "ok" => (Icon::Check, egui::Color32::LIGHT_GREEN),
                                "tool" => (Icon::Terminal, egui::Color32::LIGHT_BLUE),
                                _ => (Icon::ArrowRight, egui::Color32::GRAY),
                            };
                            ui.horizontal(|ui| {
                                ui.label(li(ic).small().color(color));
                                ui.label(egui::RichText::new(&a.text).small().color(color));
                            });
                        }
                    });
            }
        });

        // -- providers modal --------------------------------------------------
        if self.show_providers {
            let mut open = self.show_providers;
            egui::Window::new("Providers & API keys")
                .open(&mut open)
                .resizable(true)
                .default_width(520.0)
                .show(&ctx, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Keys live in Windows Credential Manager — never in the DB or logs. \
                             Ollama / local servers need no key.",
                        )
                        .small()
                        .weak(),
                    );
                    ui.separator();
                    let mut test: Option<String> = None;
                    let mut delete: Option<String> = None;
                    for p in self.providers.clone() {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(&p.name).strong());
                            ui.label(format!(
                                "{} · {}",
                                kind_label(&p.kind),
                                p.default_model.as_deref().unwrap_or("no default model")
                            ));
                            let keyed = self.keys.get(&p.name).copied().unwrap_or(false);
                            ui.horizontal(|ui| {
                                if keyed {
                                    ui.label(li(Icon::KeyRound).color(egui::Color32::LIGHT_GREEN));
                                    ui.label("key");
                                } else {
                                    ui.label(li(Icon::TriangleAlert).color(egui::Color32::GOLD));
                                    ui.label("no key");
                                }
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .button(li(Icon::Trash2))
                                        .on_hover_text("Delete provider")
                                        .clicked()
                                    {
                                        delete = Some(p.name.clone());
                                    }
                                    if ui
                                        .button(li(Icon::FlaskConical))
                                        .on_hover_text("Test connection")
                                        .clicked()
                                    {
                                        test = Some(p.name.clone());
                                    }
                                },
                            );
                        });
                    }
                    if self.providers.is_empty() {
                        ui.label(egui::RichText::new("No providers yet").weak());
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label(li(Icon::Plus).strong());
                        ui.heading("Add / update provider");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.prov_name)
                                .hint_text("e.g. my-mistral")
                                .desired_width(200.0),
                        );
                        ui.label("Kind");
                        egui::ComboBox::from_id_salt("prov_kind")
                            .selected_text(&self.prov_kind)
                            .show_ui(ui, |ui| {
                                for k in KINDS {
                                    ui.selectable_value(
                                        &mut self.prov_kind,
                                        k.to_string(),
                                        kind_label(k),
                                    );
                                }
                            });
                    });
                    ui.label(
                        egui::RichText::new(kind_blurb(&self.prov_kind))
                            .small()
                            .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Base URL");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.prov_base)
                                .hint_text(base_placeholder(&self.prov_kind))
                                .desired_width(300.0),
                        );
                        if default_base_url(&self.prov_kind).is_some()
                            && ui.button("Fill default").clicked()
                        {
                            self.prov_base = default_base_url(&self.prov_kind)
                                .unwrap_or_default()
                                .to_string();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Model");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.prov_model)
                                .hint_text("default model id")
                                .desired_width(200.0),
                        );
                    });
                    if !model_hints(&self.prov_kind).is_empty() {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(egui::RichText::new("try:").small().weak());
                            let mut pick: Option<String> = None;
                            for h in model_hints(&self.prov_kind) {
                                if ui.button(*h).clicked() {
                                    pick = Some(h.to_string());
                                }
                            }
                            if let Some(h) = pick {
                                self.prov_model = h;
                            }
                        });
                    }
                    egui::CollapsingHeader::new("Model catalog").show(ui, |ui| {
                        for m in known_models(&self.prov_kind) {
                            ui.horizontal(|ui| {
                                ui.label(li(Icon::Cpu).small().weak());
                                ui.label(egui::RichText::new(&m.display_name).strong());
                                ui.label(egui::RichText::new(&m.id).small().monospace());
                            });
                            ui.label(egui::RichText::new(catalog_line(&m)).small().weak());
                        }
                        if known_models(&self.prov_kind).is_empty() {
                            ui.label(
                                egui::RichText::new(
                                    "No offline catalog — use an id from the server's /v1/models.",
                                )
                                .small()
                                .weak(),
                            );
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("API key");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.prov_key)
                                .password(true)
                                .hint_text(if key_required(&self.prov_kind) {
                                    "required (leave empty to keep existing)"
                                } else {
                                    "optional for local servers"
                                })
                                .desired_width(280.0),
                        );
                        if ui.button("Save provider").clicked() {
                            self.save_provider();
                        }
                    });
                    if let Some(r) = self.prov_result.clone() {
                        ui.label(egui::RichText::new(r).small());
                    }
                    if let Some(name) = test {
                        self.test_provider(name);
                    }
                    if let Some(name) = delete {
                        self.delete_provider(name);
                    }
                });
            self.show_providers = open;
        }

        // -- guided model browser -------------------------------------------
        if self.show_models {
            let mut open = self.show_models;
            egui::Window::new("Choose a model")
                .open(&mut open)
                .resizable(true)
                .default_width(580.0)
                .show(&ctx, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Friendly names, prices and context. Live list when the server \
                             is reachable, built-in catalog otherwise. You can switch models \
                             any time — even mid-conversation; history is kept.",
                        )
                        .small()
                        .weak(),
                    );
                    ui.horizontal(|ui| {
                        ui.label("Provider");
                        let before = self.browser_provider.clone();
                        egui::ComboBox::from_id_salt("browser_provider")
                            .selected_text(if self.browser_provider.is_empty() {
                                "pick…"
                            } else {
                                &self.browser_provider
                            })
                            .show_ui(ui, |ui| {
                                for p in &self.providers {
                                    ui.selectable_value(
                                        &mut self.browser_provider,
                                        p.name.clone(),
                                        format!("{} ({})", p.name, kind_label(&p.kind)),
                                    );
                                }
                            });
                        if self.browser_provider != before {
                            self.reload_browser();
                        }
                        if ui
                            .button(li(Icon::RefreshCw))
                            .on_hover_text("Reload live model list")
                            .clicked()
                        {
                            self.reload_browser();
                        }
                    });
                    if self.browser_provider.is_empty() {
                        ui.label("Add a provider first (Providers & keys).");
                    } else {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut self.browser_custom)
                                    .hint_text("Or type any model id…")
                                    .desired_width(300.0),
                            );
                            if ui.button("Use custom id").clicked()
                                && !self.browser_custom.trim().is_empty()
                            {
                                self.model_sel = self.browser_custom.trim().to_string();
                                self.provider_sel = self.browser_provider.clone();
                                self.browser_custom.clear();
                                self.show_models = false;
                            }
                        });
                        if self.browser_loading {
                            ui.horizontal(|ui| {
                                ui.spinner();
                                ui.label("Loading live model list…");
                            });
                        }
                        let mut pick: Option<String> = None;
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                for m in self.browser_models.clone() {
                                    ui.horizontal(|ui| {
                                        ui.label(li(Icon::Cpu).small().weak());
                                        if ui.button(&m.display_name).clicked() {
                                            pick = Some(m.id.clone());
                                        }
                                        ui.label(egui::RichText::new(&m.id).small().monospace());
                                    });
                                    ui.label(egui::RichText::new(catalog_line(&m)).small().weak());
                                }
                                if !self.browser_loading && self.browser_models.is_empty() {
                                    ui.label(
                                        egui::RichText::new(
                                            "No models found. Check the provider setup, \
                                             or type a custom id above.",
                                        )
                                        .weak(),
                                    );
                                }
                            });
                        if let Some(id) = pick {
                            // Picking rebinds instantly (provider included);
                            // the header auto-applies on the next frame.
                            self.model_sel = id;
                            self.provider_sel = self.browser_provider.clone();
                            self.show_models = false;
                        }
                    }
                });
            self.show_models = open;
        }

        // -- diagnostics modal ------------------------------------------------
        if self.show_diags {
            let mut open = self.show_diags;
            egui::Window::new("Diagnostics")
                .open(&mut open)
                .resizable(true)
                .default_width(560.0)
                .show(&ctx, |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Self-test for the whole pipeline. No tokens spent, nothing modified. \
                             If the AI answers but never acts, check the model supports tool \
                             calling (small local models often don't).",
                        )
                        .small()
                        .weak(),
                    );
                    ui.separator();
                    if self.diags_running && self.diags.is_empty() {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Running checks…");
                        });
                    }
                    for row in self.diags.clone() {
                        ui.horizontal(|ui| {
                            let (ic, color) = match row.ok {
                                Some(true) => (Icon::Check, egui::Color32::LIGHT_GREEN),
                                Some(false) => (Icon::X, egui::Color32::LIGHT_RED),
                                None => (Icon::Minus, egui::Color32::GRAY),
                            };
                            ui.label(li(ic).color(color));
                            ui.label(egui::RichText::new(&row.name).strong());
                        });
                        ui.label(egui::RichText::new(&row.detail).small().weak());
                    }
                    ui.separator();
                    ui.horizontal(|ui| {
                        if ui.button("Re-run").clicked() {
                            self.run_diagnostics();
                        }
                        if ui.button("Close").clicked() {
                            self.show_diags = false;
                        }
                    });
                });
            self.show_diags = open;
        }

        if self.send_requested {
            self.send_requested = false;
            self.do_send();
        }
        // Poll the background channel: repaint immediately while streaming,
        // otherwise at ~8 Hz so approvals/activity land promptly.
        if self.streaming {
            ctx.request_repaint();
        } else {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
    }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

fn data_dir() -> PathBuf {
    if let Ok(roaming) = std::env::var("APPDATA") {
        PathBuf::from(roaming).join(APP_ID)
    } else {
        PathBuf::from(".").join("super-ai-data")
    }
}

/// Home for sessions without an explicit folder: `%USERPROFILE%\Super-AI`.
/// Tools always have a confined root, so file and terminal commands work
/// out of the box instead of failing with "bind a workspace first".
fn default_workspace_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| data_dir())
        .join("Super-AI")
}

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime builds");

    let data_dir = data_dir();
    std::fs::create_dir_all(&data_dir).expect("data dir is writable");
    let db_path = data_dir.join("super-ai.db");
    let db: Db = rt
        .block_on(Db::open(&db_path))
        .expect("database opens and migrates");
    let handle = rt.handle().clone();

    let policy = PolicyEngine::new(RiskClass::Low);
    let (bus_tx, _bus_rx) = broadcast::channel::<AgentEvent>(4096);
    let agent = Agent::new(policy.clone(), bus_tx.clone());

    // Restore the persisted operating mode (Plan/Build).
    let saved_mode = rt
        .block_on(db.get_setting("agent_mode"))
        .ok()
        .flatten()
        .and_then(|s| AgentMode::parse(&s))
        .unwrap_or_default();
    rt.block_on(policy.set_mode(saved_mode));

    // Single global forwarder: every agent event is persisted (flight
    // recorder) exactly once, then relayed to the UI thread.
    let (ui_tx, ui_rx) = mpsc::channel::<UiUpdate>();
    {
        let db = db.clone();
        let ui_tx = ui_tx.clone();
        let mut rx = bus_tx.subscribe();
        handle.spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(ev) => {
                        persist_event(&db, &ev).await;
                        if ui_tx.send(UiUpdate::Agent(ev)).is_err() {
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });
    }

    // Keep the runtime alive for the app's lifetime.
    std::mem::forget(rt);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Super-AI")
            .with_inner_size([1280.0, 840.0])
            .with_min_inner_size([960.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Super-AI",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_fonts(app_fonts());
            let mut visuals = egui::Visuals::dark();
            visuals.hyperlink_color = egui::Color32::from_rgb(96, 165, 250);
            cc.egui_ctx.set_visuals(visuals);
            Ok(Box::new(SuperAiApp::new(
                db, policy, agent, handle, ui_tx, ui_rx, saved_mode,
            )))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Lucide glyph the UI renders must exist in the bundled font.
    /// (Blank icons + epaint "replacement character" warnings mean a
    /// codepoint is missing here.)
    #[test]
    fn lucide_glyphs_cover_used_icons() {
        use skrifa::MetadataProvider;
        let font = skrifa::FontRef::new(lucide_icons::LUCIDE_FONT_BYTES)
            .expect("bundled lucide.ttf parses");
        let cmap = font.charmap();
        let used = [
            Icon::Bot,
            Icon::MessageSquare,
            Icon::Plus,
            Icon::X,
            Icon::SlidersHorizontal,
            Icon::Settings,
            Icon::FlaskConical,
            Icon::Trash2,
            Icon::KeyRound,
            Icon::TriangleAlert,
            Icon::User,
            Icon::ShieldAlert,
            Icon::Terminal,
            Icon::ArrowRight,
            Icon::Check,
            Icon::Activity,
            Icon::Cpu,
            Icon::Send,
            Icon::RefreshCw,
            Icon::Brain,
            Icon::Mic,
            Icon::Square,
            Icon::Stethoscope,
            Icon::Minus,
        ];
        let mut missing = Vec::new();
        for ic in used {
            let ch = char::from(ic);
            let covered = cmap.map(ch as u32).is_some_and(|g| g.to_u32() != 0);
            if !covered {
                missing.push(format!("{ic:?} U+{:04X}", ch as u32));
            }
        }
        assert!(missing.is_empty(), "icons missing glyphs: {missing:?}");
    }

    /// Same check through epaint's own font stack with the exact production
    /// `FontDefinitions`: catches registration bugs (wrong family name,
    /// unloadable bytes) that a raw cmap probe cannot see.
    #[test]
    fn epaint_resolves_used_icons() {
        use egui::epaint::text::{Fonts, TextOptions};
        let mut fonts = Fonts::new(TextOptions::default(), app_fonts());
        let view = fonts.with_pixels_per_point(1.0);
        let mut view = view;
        eprintln!(
            "font_data: {:?}",
            view.definitions().font_data.keys().collect::<Vec<_>>()
        );
        eprintln!("families: {:?}", view.families());
        let id = egui::FontId::new(14.0, egui::FontFamily::Name("lucide".into()));
        let used = [
            Icon::Bot,
            Icon::MessageSquare,
            Icon::Plus,
            Icon::X,
            Icon::SlidersHorizontal,
            Icon::Settings,
            Icon::FlaskConical,
            Icon::Trash2,
            Icon::KeyRound,
            Icon::TriangleAlert,
            Icon::User,
            Icon::ShieldAlert,
            Icon::Terminal,
            Icon::ArrowRight,
            Icon::Check,
            Icon::Activity,
            Icon::Cpu,
            Icon::Send,
            Icon::RefreshCw,
            Icon::Brain,
            Icon::Mic,
            Icon::Square,
            Icon::Stethoscope,
            Icon::Minus,
        ];
        let mut missing = Vec::new();
        for ic in used {
            let ch = char::from(ic);
            if !view.has_glyph(&id, ch) {
                missing.push(format!("{ic:?} U+{:04X}", ch as u32));
            }
        }
        assert!(missing.is_empty(), "epaint can't resolve: {missing:?}");

        // Full layout path (shaping + atlas): a resolved glyph must also
        // produce a non-empty galley, or it renders blank at runtime.
        use egui::epaint::text::LayoutJob;
        let mut blank = Vec::new();
        for ic in used {
            let job = LayoutJob::simple(
                char::from(ic).to_string(),
                id.clone(),
                egui::Color32::WHITE,
                200.0,
            );
            let galley = view.layout_job(job);
            if galley.size().x <= 0.0 {
                blank.push(format!("{ic:?}"));
            }
        }
        assert!(blank.is_empty(), "icons lay out blank: {blank:?}");

        // The Hack fallback resolves the replacement char, so epaint never
        // logs its "replacement character" warning for this family. (Note:
        // `has_glyph('?')` is documented to return false for the replacement
        // char itself, so we assert resolution indirectly: every icon above
        // resolved to a NON-replacement face, which requires the family
        // chain — lucide first, Hack second — to be intact.)
        let families = view.families();
        assert!(
            families.contains(&egui::FontFamily::Name("lucide".into())),
            "lucide family must be registered"
        );
    }
}
