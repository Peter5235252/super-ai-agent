//! Typed Tauri commands — the only surface the webview can touch.
//!
//! Mirrors the native shell one-to-one: sessions, BYOK providers (7 kinds),
//! test-with-ping-fallback, model catalogs, send with tools + modes +
//! reasoning effort, approvals, diagnostics, settings. Secrets stay in the
//! Rust process / OS credential store — a raw API key never crosses IPC.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::Arc;

use agent_runtime::{TaskRequest, ToolRegistry};
use app_core::{AgentMode, ApprovalDecision, ReasoningEffort, RiskClass, WorkspaceConfig, now_ms};
use futures::StreamExt;
use persistence::{Db, MessageRow, ProviderRow, SessionRow};
use provider_api::{Message, MessageRole, ModelInfo, ModelProvider};
use secrecy::SecretString;
use tauri::State;
use tool_core::{Tool, ToolContext};

use super::{AppState, build_provider};

const SERVICE: &str = "super-ai";

fn err<E: std::fmt::Display>(e: E) -> String {
    e.to_string()
}

fn parse_uuid(s: &str) -> Result<uuid::Uuid, String> {
    uuid::Uuid::parse_str(s).map_err(|e| format!("invalid id: {e}"))
}

fn none_if_empty(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

fn key_required(kind: &str) -> bool {
    matches!(kind, "openai" | "anthropic" | "xai" | "mistral" | "gemini")
}

fn stored_key(name: &str) -> Result<Option<SecretString>, String> {
    secrets::SecretStore::get(SERVICE, name).map_err(err)
}

fn require_key(name: &str) -> Result<SecretString, String> {
    match stored_key(name)? {
        Some(k) => Ok(k),
        None => Err(format!(
            "no key stored for '{name}' — add it in Providers & keys"
        )),
    }
}

/// Home for sessions without an explicit folder (`%USERPROFILE%\Super-AI`),
/// mirroring the native shell so tools always have a confined root.
fn default_workspace_dir() -> PathBuf {
    std::env::var("USERPROFILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("Super-AI")
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn create_session(
    state: State<'_, AppState>,
    title: Option<String>,
    workspace: Option<String>,
) -> Result<SessionRow, String> {
    let id = uuid::Uuid::new_v4();
    let ws = workspace.unwrap_or_default();
    let ws_path = if ws.trim().is_empty() {
        let dir = default_workspace_dir();
        let _ = std::fs::create_dir_all(&dir);
        dir.to_string_lossy().to_string()
    } else {
        ws
    };
    state
        .db
        .create_session(
            id,
            title.as_deref().unwrap_or("New session"),
            Some(&ws_path),
        )
        .await
        .map_err(err)
}

#[tauri::command]
pub async fn list_sessions(state: State<'_, AppState>) -> Result<Vec<SessionRow>, String> {
    state.db.list_sessions().await.map_err(err)
}

#[tauri::command]
pub async fn delete_session(state: State<'_, AppState>, id: String) -> Result<(), String> {
    state.db.delete_session(parse_uuid(&id)?).await.map_err(err)
}

#[tauri::command]
pub async fn get_messages(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<Vec<MessageRow>, String> {
    state
        .db
        .list_messages(parse_uuid(&session_id)?)
        .await
        .map_err(err)
}

#[tauri::command]
pub async fn bind_session_provider(
    state: State<'_, AppState>,
    session_id: String,
    provider_name: String,
    model: String,
) -> Result<(), String> {
    if model.trim().is_empty() {
        return Err("model is required".into());
    }
    state
        .db
        .bind_provider(parse_uuid(&session_id)?, &provider_name, &model)
        .await
        .map_err(err)
}

#[tauri::command]
pub async fn set_session_effort(
    state: State<'_, AppState>,
    session_id: String,
    effort: Option<String>,
) -> Result<(), String> {
    if let Some(ref e) = effort {
        ReasoningEffort::parse(e).ok_or_else(|| format!("unknown effort: {e}"))?;
    }
    state
        .db
        .set_session_effort(parse_uuid(&session_id)?, effort.as_deref())
        .await
        .map_err(err)
}

#[tauri::command]
pub async fn set_mode(state: State<'_, AppState>, mode: AgentMode) -> Result<String, String> {
    state.policy.set_mode(mode).await;
    state
        .db
        .set_setting("agent_mode", mode.as_str())
        .await
        .map_err(err)?;
    Ok(match mode {
        AgentMode::Build => "Build mode — the agent can act (approvals still apply)".into(),
        AgentMode::Plan => "Plan mode — read-only".into(),
    })
}

#[tauri::command]
pub async fn set_autonomy(state: State<'_, AppState>, risk: RiskClass) -> Result<(), String> {
    state.policy.set_auto_approve(risk).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Providers (BYOK)
// ---------------------------------------------------------------------------

fn validate_provider_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("provider name is required".into());
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err("provider name may only contain letters, digits, '-', '_', '.'".into());
    }
    Ok(())
}

#[tauri::command]
pub async fn list_providers(state: State<'_, AppState>) -> Result<Vec<ProviderRow>, String> {
    state.db.list_providers().await.map_err(err)
}

#[tauri::command]
pub async fn save_provider(
    state: State<'_, AppState>,
    name: String,
    kind: String,
    base_url: Option<String>,
    default_model: Option<String>,
    api_key: Option<String>,
) -> Result<String, String> {
    validate_provider_name(&name)?;
    match kind.as_str() {
        "openai" | "anthropic" | "xai" | "mistral" | "gemini" | "ollama" | "local" => {}
        other => return Err(format!("unsupported provider kind: {other}")),
    }
    // Never store the key in SQLite; hand it straight to the OS store.
    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
        let secret = SecretString::new(key.into());
        secrets::SecretStore::set(SERVICE, &name, &secret).map_err(err)?;
    }
    let row = ProviderRow {
        name: name.clone(),
        kind,
        base_url,
        default_model,
        is_default: false,
        created_at: now_ms(),
        updated_at: now_ms(),
    };
    state.db.upsert_provider(&row).await.map_err(err)?;
    Ok(format!(
        "provider '{name}' saved (key in Windows Credential Manager)"
    ))
}

#[tauri::command]
pub async fn delete_provider(state: State<'_, AppState>, name: String) -> Result<String, String> {
    let _ = secrets::SecretStore::delete(SERVICE, &name);
    state.db.delete_provider(&name).await.map_err(err)?;
    Ok(format!("provider '{name}' deleted"))
}

/// Whether a key is stored for this provider — presence only, never the value.
#[tauri::command]
pub async fn has_provider_key(name: String) -> Result<bool, String> {
    Ok(secrets::SecretStore::get(SERVICE, &name)
        .map_err(err)?
        .is_some())
}

async fn provider_for(
    db: &Db,
    name: &str,
) -> Result<(Arc<dyn ModelProvider>, ProviderRow), String> {
    let row = db
        .get_provider(name)
        .await
        .map_err(err)?
        .ok_or_else(|| format!("provider '{name}' not found"))?;
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

#[tauri::command]
pub async fn test_provider(state: State<'_, AppState>, name: String) -> Result<String, String> {
    let (provider, row) = provider_for(&state.db, &name).await?;
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
            // Some compat servers skip /models: fall back to a tiny ping.
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

/// Live model list for one configured provider (falls back to
/// [`known_models`] offline data when the server is unreachable).
#[tauri::command]
pub async fn live_models(
    state: State<'_, AppState>,
    name: String,
) -> Result<Vec<ModelInfo>, String> {
    let (provider, _) = provider_for(&state.db, &name).await?;
    provider.list_models().await.map_err(err)
}

/// Model catalog for one provider family (offline metadata, no API call).
#[tauri::command]
pub async fn known_models(kind: String) -> Result<Vec<ModelInfo>, String> {
    let models: Vec<ModelInfo> = match kind.as_str() {
        "openai" => provider_openai::models::known_models(),
        "anthropic" => provider_anthropic::models::known_models(),
        "xai" => provider_xai::models::known_models(),
        "mistral" => provider_compat::models::known_models_mistral(),
        "gemini" => provider_compat::models::known_models_gemini(),
        "ollama" => provider_compat::models::known_models_ollama(),
        "local" => provider_compat::models::known_models_local(),
        other => return Err(format!("unsupported provider kind: {other}")),
    };
    Ok(models)
}

// ---------------------------------------------------------------------------
// Agent tasks
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    session_id: String,
    text: String,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("message is empty".into());
    }
    let session_id = parse_uuid(&session_id)?;
    let session = state
        .db
        .get_session(session_id)
        .await
        .map_err(err)?
        .ok_or_else(|| "session not found".to_string())?;

    let provider_name = session
        .provider_name
        .clone()
        .ok_or_else(|| "no provider set for this session — pick one in the header".to_string())?;
    let row = state
        .db
        .get_provider(&provider_name)
        .await
        .map_err(err)?
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

    let rows = state.db.list_messages(session_id).await.map_err(err)?;
    let history: Vec<Message> = rows.iter().filter_map(to_internal_message).collect();

    // Every session resolves to a confined root (bound folder or the
    // ~/Super-AI fallback), so file + terminal tools always work.
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

    state
        .db
        .insert_message(&MessageRow {
            id: uuid::Uuid::new_v4(),
            session_id,
            role: "user".into(),
            content: text.clone(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            created_at: now_ms(),
        })
        .await
        .map_err(err)?;
    state.db.touch_session(session_id).await.map_err(err)?;

    let mut system = app_core::SYSTEM_PROMPT.to_string();
    if state.policy.mode().await == app_core::AgentMode::Build {
        system.push_str(
            "\n\nCurrent mode: BUILD — use any tool you need. File writes and terminal \
             commands need human approval and may be denied; if so, report it and adapt.",
        );
    } else {
        system.push_str(
            "\n\nCurrent mode: PLAN (read-only) — ONLY use fs.list and fs.read to investigate, \
             then explain your plan step by step. Do NOT call fs.write or process.run: they \
             are blocked in this mode. End by asking the user to switch to Build mode if \
             they want you to implement it.",
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
        .and_then(provider_api::ReasoningEffort::parse);

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

    // Supervised inside the agent (panic reporting, abort registry); the
    // outcome arrives via the event bus.
    state.agent.spawn(request);
    Ok("task started".into())
}

#[tauri::command]
pub async fn approve(
    state: State<'_, AppState>,
    approval_id: String,
    allow: bool,
) -> Result<bool, String> {
    let id = parse_uuid(&approval_id)?;
    let decision = if allow {
        ApprovalDecision::AllowOnce
    } else {
        ApprovalDecision::Deny
    };
    Ok(state.policy.respond(id, decision).await)
}

// ---------------------------------------------------------------------------
// Diagnostics (no tokens spent, nothing modified)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiagRow {
    pub name: String,
    /// None = skipped.
    pub ok: Option<bool>,
    pub detail: String,
}

#[tauri::command]
pub async fn run_diagnostics(
    state: State<'_, AppState>,
    session_id: Option<String>,
) -> Result<Vec<DiagRow>, String> {
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
        match state.db.list_sessions().await {
            Ok(s) => Ok(format!("sessions table readable ({} sessions)", s.len())),
            Err(e) => Err(format!("unreadable: {e}")),
        },
    );

    let dir = default_workspace_dir();
    row(
        "Workspace folder",
        match std::fs::create_dir_all(&dir).and_then(|_| std::fs::read_dir(&dir).map(|_| ())) {
            Ok(()) => Ok(format!("usable: {}", dir.display())),
            Err(e) => Err(format!("can't use {}: {e}", dir.display())),
        },
    );

    row("Approval gate", {
        let policy = policy_engine::PolicyEngine::new(RiskClass::Low);
        let def = tool_core::ToolDefinition::new(
            "diag.probe",
            "diagnostic probe (never executed)",
            serde_json::json!({"type": "object"}),
            RiskClass::Medium,
            vec![app_core::SideEffect::WritesFiles],
        );
        match policy
            .authorize(&def, &serde_json::Value::Null, &ToolContext::default())
            .await
        {
            policy_engine::Decision::PendingApproval { id, receiver, .. } => {
                policy.respond(id, ApprovalDecision::AllowOnce).await;
                match receiver.await {
                    Ok(ApprovalDecision::AllowOnce) => {
                        Ok("raise → answer → resolve round-trip works".to_string())
                    }
                    Ok(other) => Err(format!("gate answered wrong: {other:?}")),
                    Err(_) => Err("approval answer got lost".to_string()),
                }
            }
            policy_engine::Decision::Allow => {
                Err("expected an approval gate, got auto-allow".to_string())
            }
            policy_engine::Decision::Deny { reason } => Err(format!("unexpected deny: {reason}")),
        }
    });

    row("File tools (list)", {
        let tools = tool_filesystem::filesystem_tools(dir.clone());
        match tools.into_iter().find(|t| t.definition().name == "fs.list") {
            None => Err("fs.list tool missing".to_string()),
            Some(list) => {
                let ctx = ToolContext {
                    workspace: Some(WorkspaceConfig::from_dir("diag", dir.clone())),
                    cwd: Some(dir.clone()),
                };
                match list.execute(serde_json::json!({"path": "."}), &ctx).await {
                    Ok(out) => Ok(format!("fs.list works ({} chars listed)", out.output.len())),
                    Err(e) => Err(format!("{e:#}")),
                }
            }
        }
    });

    row("Terminal tool", {
        let tools = tool_process::process_tools();
        match tools
            .into_iter()
            .find(|t| t.definition().name == "process.run")
        {
            None => Err("process.run tool missing".to_string()),
            Some(run) => {
                let ctx = ToolContext {
                    workspace: Some(WorkspaceConfig::from_dir("diag", dir.clone())),
                    cwd: Some(dir.clone()),
                };
                let fut = run.execute(
                    serde_json::json!({"shell": "powershell", "command": "Write-Output diag-ok", "timeout_ms": 15000}),
                    &ctx,
                );
                match tokio::time::timeout(std::time::Duration::from_secs(25), fut).await {
                    Err(_) => Err("terminal call timed out".to_string()),
                    Ok(Err(e)) => Err(format!("{e:#}")),
                    Ok(Ok(out)) => {
                        if out.output.contains("diag-ok") && out.output.contains("exit: 0") {
                            Ok("PowerShell round-trip works".to_string())
                        } else {
                            Err(format!(
                                "unexpected output: {}",
                                out.output.chars().take(200).collect::<String>()
                            ))
                        }
                    }
                }
            }
        }
    });

    let provider_name = match session_id {
        Some(sid) => {
            let sid = parse_uuid(&sid)?;
            state
                .db
                .get_session(sid)
                .await
                .map_err(err)?
                .and_then(|s| s.provider_name)
        }
        None => None,
    };
    let provider_name = match provider_name {
        Some(n) => Some(n),
        None => state
            .db
            .list_providers()
            .await
            .map_err(err)?
            .first()
            .map(|p| p.name.clone()),
    };
    match provider_name {
        Some(name) => {
            let label = format!("Model connection ({name})");
            match provider_for(&state.db, &name).await {
                Ok((provider, _)) => match provider.list_models().await {
                    Ok(models) => row(
                        &label,
                        Ok(format!(
                            "{} live models{}",
                            models.len(),
                            models
                                .first()
                                .map(|m| format!(" (e.g. {})", m.id))
                                .unwrap_or_default()
                        )),
                    ),
                    Err(e) => row(&label, Err(format!("unreachable: {e}"))),
                },
                Err(e) => row(&label, Err(e)),
            }
        }
        None => rows.push(DiagRow {
            name: "Model connection".to_string(),
            ok: None,
            detail: "skipped: add a provider first".to_string(),
        }),
    }

    Ok(rows)
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_setting(
    state: State<'_, AppState>,
    key: String,
) -> Result<Option<String>, String> {
    state.db.get_setting(&key).await.map_err(err)
}

#[tauri::command]
pub async fn set_setting(
    state: State<'_, AppState>,
    key: String,
    value: String,
) -> Result<(), String> {
    state.db.set_setting(&key, &value).await.map_err(err)
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
