//! Super-AI web shell (Tauri 2) — lives on the `tauri-shell` branch.
//!
//! Same Rust backend as the native shell: the webview is a thin renderer
//! talking through typed commands, with agent events streamed over the
//! `agent-event` channel. Sessions, keys and history are shared with the
//! native app (same SQLite file, same credential store).
#![forbid(unsafe_code)]

mod commands;

use std::path::PathBuf;
use std::sync::Arc;

use agent_runtime::{Agent, AgentEvent};
use app_core::RiskClass;
use persistence::Db;
use policy_engine::PolicyEngine;
use provider_api::ModelProvider;
use secrecy::SecretString;
use tauri::{AppHandle, Emitter, Manager};
use tokio::sync::broadcast;
use tracing_subscriber::EnvFilter;

pub struct AppState {
    pub db: Db,
    pub policy: Arc<PolicyEngine>,
    pub agent: Arc<Agent>,
}

impl AppState {
    async fn new(db_path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let db = Db::open(&db_path).await?;
        let policy = PolicyEngine::new(RiskClass::Low);
        let (tx, _rx) = broadcast::channel(4096);
        let agent = Agent::new(policy.clone(), tx);
        Ok(Self { db, policy, agent })
    }
}

/// Construct a provider adapter from stored configuration + key.
/// Local kinds (`ollama`, `local`) work without a key.
pub fn build_provider(
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

/// Persist + forward one agent event to the webview.
/// Token deltas skip the flight recorder (UI-live only); everything else
/// is stored, and conversation turns are mirrored into `messages` with
/// their tool calls so reloaded history stays provider-valid.
pub async fn forward_event(app: &AppHandle, db: &Db, ev: &AgentEvent) {
    if matches!(
        ev,
        AgentEvent::ModelDelta { .. } | AgentEvent::ReasoningDelta { .. }
    ) {
        let payload = match serde_json::to_value(ev) {
            Ok(p) => p,
            Err(_) => return,
        };
        let _ = app.emit("agent-event", payload);
        return;
    }
    let payload = match serde_json::to_value(ev) {
        Ok(p) => p,
        Err(_) => return,
    };
    let _ = app.emit("agent-event", payload.clone());

    let (task_id, session_id) = event_ids(ev);
    let _ = db
        .insert_event(task_id, session_id, event_kind(ev), &payload)
        .await;

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
                .insert_message(&persistence::MessageRow {
                    id: uuid::Uuid::new_v4(),
                    session_id: *session_id,
                    role: "assistant".into(),
                    content: text.clone(),
                    tool_calls,
                    tool_call_id: None,
                    reasoning,
                    created_at: app_core::now_ms(),
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
                .insert_message(&persistence::MessageRow {
                    id: uuid::Uuid::new_v4(),
                    session_id: *session_id,
                    role: "tool".into(),
                    content: output.clone(),
                    tool_calls: None,
                    tool_call_id: Some(tool_call_id.clone()),
                    reasoning: None,
                    created_at: app_core::now_ms(),
                })
                .await;
        }
        _ => {}
    }

    if let Some(sid) = session_id {
        let _ = db.touch_session(sid).await;
    }
}

fn event_ids(ev: &AgentEvent) -> (Option<app_core::TaskId>, Option<app_core::SessionId>) {
    use agent_runtime::AgentEvent::*;
    match ev {
        TaskCreated {
            task_id,
            session_id,
            ..
        }
        | ReasoningStarted {
            task_id,
            session_id,
        }
        | ModelDelta {
            task_id,
            session_id,
            ..
        }
        | ReasoningDelta {
            task_id,
            session_id,
            ..
        }
        | ToolRequested {
            task_id,
            session_id,
            ..
        }
        | ToolStarted {
            task_id,
            session_id,
            ..
        }
        | ToolOutput {
            task_id,
            session_id,
            ..
        }
        | ApprovalRequested {
            task_id,
            session_id,
            ..
        }
        | ApprovalResolved {
            task_id,
            session_id,
            ..
        }
        | MessageCompleted {
            task_id,
            session_id,
            ..
        }
        | TaskFailed {
            task_id,
            session_id,
            ..
        }
        | TaskCompleted {
            task_id,
            session_id,
            ..
        } => (Some(*task_id), Some(*session_id)),
    }
}

fn event_kind(ev: &AgentEvent) -> &'static str {
    use agent_runtime::AgentEvent::*;
    match ev {
        TaskCreated { .. } => "task_created",
        ReasoningStarted { .. } => "reasoning_started",
        ModelDelta { .. } => "model_delta",
        ReasoningDelta { .. } => "reasoning_delta",
        ToolRequested { .. } => "tool_requested",
        ToolStarted { .. } => "tool_started",
        ToolOutput { .. } => "tool_output",
        ApprovalRequested { .. } => "approval_requested",
        ApprovalResolved { .. } => "approval_resolved",
        MessageCompleted { .. } => "message_completed",
        TaskFailed { .. } => "task_failed",
        TaskCompleted { .. } => "task_completed",
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    tauri::Builder::default()
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(&data_dir)?;
            let db_path = data_dir.join("super-ai.db");
            let state = tauri::async_runtime::block_on(AppState::new(db_path))?;
            tauri::async_runtime::block_on(async {
                if let Ok(Some(saved)) = state.db.get_setting("agent_mode").await {
                    if let Some(mode) = app_core::AgentMode::parse(&saved) {
                        state.policy.set_mode(mode).await;
                    }
                }
            });

            // Single global forwarder: every agent event reaches the webview
            // and the flight-recorder table exactly once, regardless of how
            // many tasks run concurrently.
            let db = state.db.clone();
            let mut rx = state.agent.events.subscribe();
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    match rx.recv().await {
                        Ok(ev) => forward_event(&handle, &db, &ev).await,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            });

            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::create_session,
            commands::list_sessions,
            commands::delete_session,
            commands::get_messages,
            commands::bind_session_provider,
            commands::set_session_effort,
            commands::set_mode,
            commands::set_autonomy,
            commands::list_providers,
            commands::save_provider,
            commands::delete_provider,
            commands::has_provider_key,
            commands::test_provider,
            commands::known_models,
            commands::live_models,
            commands::send_message,
            commands::approve,
            commands::run_diagnostics,
            commands::get_setting,
            commands::set_setting,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Super-AI");
}
