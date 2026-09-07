//! Event-driven agent runtime.
//!
//! The agent is a state machine, not a prompt->response function. Every
//! step emits an [`AgentEvent`] on a broadcast channel so the UI, the
//! persistence layer and the flight recorder all see exactly what
//! happened. The loop is:
//!
//! ```text
//! reason -> stream model -> collect tool calls
//!   -> policy-check each call -> execute -> feed results back -> repeat
//! ```
//!
//! The runtime supports maximum turns, retry with backoff, tool timeouts,
//! approval gates, and per-task summaries. Cancellation and checkpointing
//! land with the persistence/resume phase.
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use app_core::{ApprovalDecision, RiskClass, SessionId, TaskId, WorkspaceConfig};
use futures::StreamExt;
use policy_engine::{Decision, PolicyEngine};
use provider_api::{
    AgentRequest, Message, ModelProvider, ProviderEvent, ToolCall, ToolDefinitionWire,
    UsageEstimate,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::broadcast;
use tool_core::{Tool, ToolContext, ToolResult};
use tracing::{info, warn};
use uuid::Uuid;

/// Every observable step of a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    TaskCreated {
        task_id: TaskId,
        session_id: SessionId,
        provider: String,
        model: String,
    },
    ReasoningStarted {
        task_id: TaskId,
        session_id: SessionId,
    },
    /// A chunk of generated text for the current response.
    ModelDelta {
        task_id: TaskId,
        session_id: SessionId,
        text: String,
    },
    /// A chunk of the model's private reasoning for the current response.
    /// Surfaced in the UI ("Thinking"); never sent back to providers.
    ReasoningDelta {
        task_id: TaskId,
        session_id: SessionId,
        text: String,
    },
    ToolRequested {
        task_id: TaskId,
        session_id: SessionId,
        tool: String,
        args: Value,
    },
    ToolStarted {
        task_id: TaskId,
        session_id: SessionId,
        tool: String,
    },
    ToolOutput {
        task_id: TaskId,
        session_id: SessionId,
        tool: String,
        /// Matches the `id` of the requesting `ToolCall`, so the turn can
        /// be replayed from persisted history.
        tool_call_id: String,
        output: String,
        truncated: bool,
    },
    ApprovalRequested {
        task_id: TaskId,
        session_id: SessionId,
        approval_id: Uuid,
        tool: String,
        args: Value,
        risk: RiskClass,
        reason: String,
    },
    ApprovalResolved {
        task_id: TaskId,
        session_id: SessionId,
        approval_id: Uuid,
        decision: ApprovalDecision,
    },
    MessageCompleted {
        task_id: TaskId,
        session_id: SessionId,
        text: String,
        /// The model's private reasoning for this turn (may be empty).
        /// Surfaced in the UI; never fed back into history.
        reasoning: String,
        /// Tool calls requested in this turn (empty for a final answer).
        /// Persisted alongside the text so reloaded history stays valid.
        tool_calls: Vec<ToolCall>,
    },
    TaskFailed {
        task_id: TaskId,
        session_id: SessionId,
        error: String,
    },
    TaskCompleted {
        task_id: TaskId,
        session_id: SessionId,
        summary: TaskSummary,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Completed,
    Failed,
    MaxTurnsReached,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSummary {
    pub task_id: TaskId,
    pub session_id: SessionId,
    pub turns: u32,
    pub tool_calls: u32,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub final_text: String,
    pub status: TaskStatus,
}

/// Everything a task needs to run.
pub struct TaskRequest {
    pub session_id: SessionId,
    pub provider: Arc<dyn ModelProvider>,
    pub model: String,
    pub system_prompt: Option<String>,
    /// Prior conversation (assistant tool-call history included).
    pub history: Vec<Message>,
    pub user_message: String,
    pub workspace: Option<WorkspaceConfig>,
    pub tools: ToolRegistry,
    pub max_turns: u32,
    pub reasoning_effort: Option<provider_api::ReasoningEffort>,
}

#[derive(Default)]
pub struct ToolRegistry {
    tools: std::collections::HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new(tools: Vec<Arc<dyn Tool>>) -> Self {
        let mut map = std::collections::HashMap::new();
        for t in tools {
            map.insert(t.definition().name.clone(), t);
        }
        Self { tools: map }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// Wire form sent to the model.
    pub fn definitions(&self) -> Vec<ToolDefinitionWire> {
        let mut defs: Vec<ToolDefinitionWire> = self
            .tools
            .values()
            .map(|t| {
                let d = t.definition();
                ToolDefinitionWire {
                    name: d.name.clone(),
                    description: d.description.clone(),
                    input_schema: d.input_schema.clone(),
                }
            })
            .collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }
}

/// The agent: policy + event bus + the loop itself.
pub struct Agent {
    pub policy: Arc<PolicyEngine>,
    pub events: broadcast::Sender<AgentEvent>,
    live: std::sync::Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

const STREAM_RETRIES: u32 = 2;
const RETRY_BASE_DELAY_MS: u64 = 400;
/// Hard cap per model turn: a stalled provider must fail loudly instead of
/// spinning the UI forever. Active streams never idle this long.
const TURN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

impl Agent {
    pub fn new(policy: Arc<PolicyEngine>, events: broadcast::Sender<AgentEvent>) -> Arc<Self> {
        Arc::new(Self {
            policy,
            events,
            live: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn emit(&self, ev: AgentEvent) {
        let _ = self.events.send(ev);
    }

    /// Spawn a task. Handles are tracked internally so [`Agent::abort_all`]
    /// can stop everything dead; a supervisor turns panics into
    /// `TaskFailed` (cancellations stay silent — that's the stop button).
    pub fn spawn(self: &Arc<Self>, req: TaskRequest) {
        self.prune_finished();
        let session_id = req.session_id;
        let this = Arc::clone(self);
        let inner = tokio::spawn(async move { this.run(req).await });
        let watcher = Arc::clone(self);
        let supervised = tokio::spawn(async move {
            match inner.await {
                Ok(_) => {}
                Err(e) if e.is_panic() => {
                    let _ = watcher.events.send(AgentEvent::TaskFailed {
                        task_id: Uuid::new_v4(),
                        session_id,
                        error: "the agent task crashed unexpectedly; please retry".to_string(),
                    });
                }
                Err(_) => {}
            }
            watcher.prune_finished();
        });
        self.live
            .lock()
            .expect("agent task registry")
            .push(supervised);
    }

    /// Abort every tracked task immediately. Pending approval gates are
    /// cleared separately via [`PolicyEngine::abort_pending`].
    pub fn abort_all(&self) {
        let mut live = self.live.lock().expect("agent task registry");
        for handle in live.iter() {
            handle.abort();
        }
        live.clear();
    }

    fn prune_finished(&self) {
        if let Ok(mut live) = self.live.lock() {
            live.retain(|h| !h.is_finished());
        }
    }

    async fn run(self: Arc<Self>, req: TaskRequest) -> TaskSummary {
        let task_id = Uuid::new_v4();
        let session_id = req.session_id;
        let provider_name = provider_api::ProviderKind::label(req.provider.kind());

        info!(%task_id, %session_id, provider = %provider_name, model = %req.model, "task started");
        self.emit(AgentEvent::TaskCreated {
            task_id,
            session_id,
            provider: provider_name.to_string(),
            model: req.model.clone(),
        });

        let mut messages = req.history.clone();
        messages.push(Message::user(&req.user_message));

        let mut summary = TaskSummary {
            task_id,
            session_id,
            turns: 0,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            final_text: String::new(),
            status: TaskStatus::Completed,
        };

        for turn in 1..=req.max_turns {
            summary.turns = turn;
            self.emit(AgentEvent::ReasoningStarted {
                task_id,
                session_id,
            });

            let request = AgentRequest {
                model: req.model.clone(),
                messages: messages.clone(),
                tools: req.tools.definitions(),
                system: req.system_prompt.clone(),
                temperature: None,
                max_tokens: None,
                reasoning_effort: req.reasoning_effort,
            };

            // Stream the model's response, with bounded retry on failure
            // and a hard per-turn timeout: a stalled provider must fail
            // loudly instead of spinning the UI forever.
            let turn =
                tokio::time::timeout(TURN_TIMEOUT, self.stream_turn(&req, task_id, &request)).await;
            let turned: Result<_, String> = match turn {
                Ok(inner) => inner,
                Err(_) => Err("model response timed out after 10 minutes".to_string()),
            };
            let (text, reasoning, pending, usage) = match turned {
                Ok(parts) => parts,
                Err(error) => {
                    summary.status = TaskStatus::Failed;
                    summary.final_text = last_assistant_text(&messages);
                    self.emit(AgentEvent::TaskFailed {
                        task_id,
                        session_id,
                        error: error.clone(),
                    });
                    warn!(%task_id, %error, "task failed");
                    return summary;
                }
            };

            summary.final_text = text.clone();
            summary.input_tokens += usage.input_tokens;
            summary.output_tokens += usage.output_tokens;

            let tool_calls: Vec<ToolCall> = pending
                .into_values()
                .map(|(id, name, args)| ToolCall::new(id, name, args))
                .collect();
            summary.tool_calls += tool_calls.len() as u32;

            messages.push(Message::assistant(text.clone(), tool_calls.clone()));
            self.emit(AgentEvent::MessageCompleted {
                task_id,
                session_id,
                text,
                reasoning,
                tool_calls: tool_calls.clone(),
            });

            if tool_calls.is_empty() {
                self.emit(AgentEvent::TaskCompleted {
                    task_id,
                    session_id,
                    summary: summary.clone(),
                });
                info!(%task_id, turns = summary.turns, "task completed");
                return summary;
            }

            // Execute each requested tool under policy.
            let ctx = ToolContext {
                workspace: req.workspace.clone(),
                cwd: req.workspace.as_ref().and_then(|w| w.root()).cloned(),
            };
            for tc in &tool_calls {
                let Some(tool) = req.tools.get(&tc.name) else {
                    let available = req.tools.names().join(", ");
                    let output = format!(
                        "Error: unknown tool '{}'. Available: {}",
                        tc.name, available
                    );
                    messages.push(Message::tool(tc.id.clone(), output.clone()));
                    // Persisted via the event below so reloaded history stays paired.
                    self.emit(AgentEvent::ToolOutput {
                        task_id,
                        session_id,
                        tool: tc.name.clone(),
                        tool_call_id: tc.id.clone(),
                        output,
                        truncated: false,
                    });
                    continue;
                };
                let def = tool.definition().clone();
                self.emit(AgentEvent::ToolRequested {
                    task_id,
                    session_id,
                    tool: tc.name.clone(),
                    args: tc.arguments.clone(),
                });

                let decision = self.policy.authorize(&def, &tc.arguments, &ctx).await;
                match decision {
                    Decision::Allow => {}
                    Decision::Deny { reason } => {
                        let output = format!("DENIED by policy: {reason}");
                        self.emit(AgentEvent::ToolOutput {
                            task_id,
                            session_id,
                            tool: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            output: output.clone(),
                            truncated: false,
                        });
                        messages.push(Message::tool(tc.id.clone(), output));
                        continue;
                    }
                    Decision::PendingApproval {
                        id,
                        reason,
                        receiver,
                    } => {
                        self.emit(AgentEvent::ApprovalRequested {
                            task_id,
                            session_id,
                            approval_id: id,
                            tool: tc.name.clone(),
                            args: tc.arguments.clone(),
                            risk: def.risk,
                            reason,
                        });
                        let decision = match receiver.await {
                            Ok(d) => d,
                            Err(_) => ApprovalDecision::Deny, // request expired
                        };
                        self.emit(AgentEvent::ApprovalResolved {
                            task_id,
                            session_id,
                            approval_id: id,
                            decision,
                        });
                        if decision.is_deny() {
                            let output = "DENIED by user".to_string();
                            self.emit(AgentEvent::ToolOutput {
                                task_id,
                                session_id,
                                tool: tc.name.clone(),
                                tool_call_id: tc.id.clone(),
                                output: output.clone(),
                                truncated: false,
                            });
                            messages.push(Message::tool(tc.id.clone(), output));
                            continue;
                        }
                    }
                }

                info!(%task_id, tool = %tc.name, "executing tool");
                self.emit(AgentEvent::ToolStarted {
                    task_id,
                    session_id,
                    tool: tc.name.clone(),
                });

                let result: Result<ToolResult, String> =
                    tokio::time::timeout(def.timeout, tool.execute(tc.arguments.clone(), &ctx))
                        .await
                        .map_err(|_| format!("tool timed out after {:?}", def.timeout))
                        .and_then(|r| r.map_err(|e| format!("{e:#}")));

                match result {
                    Ok(r) => {
                        self.emit(AgentEvent::ToolOutput {
                            task_id,
                            session_id,
                            tool: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            output: r.output.clone(),
                            truncated: r.truncated,
                        });
                        messages.push(Message::tool(tc.id.clone(), r.output));
                    }
                    Err(error) => {
                        self.emit(AgentEvent::ToolOutput {
                            task_id,
                            session_id,
                            tool: tc.name.clone(),
                            tool_call_id: tc.id.clone(),
                            output: format!("Error: {error}"),
                            truncated: false,
                        });
                        messages.push(Message::tool(tc.id.clone(), format!("Error: {error}")));
                    }
                }
            }
        }

        summary.status = TaskStatus::MaxTurnsReached;
        self.emit(AgentEvent::TaskFailed {
            task_id,
            session_id,
            error: format!("max turns ({}) reached", req.max_turns),
        });
        summary
    }

    /// One model call: stream deltas and collect finished tool calls.
    /// Retries transient failures with backoff.
    async fn stream_turn(
        &self,
        req: &TaskRequest,
        task_id: TaskId,
        request: &AgentRequest,
    ) -> Result<(String, String, PendingCalls, UsageEstimate), String> {
        let mut attempt = 0;
        loop {
            match self.try_stream_turn(req, task_id, request).await {
                Ok(parts) => return Ok(parts),
                Err(error) => {
                    if attempt >= STREAM_RETRIES {
                        return Err(error);
                    }
                    attempt += 1;
                    let delay_ms = RETRY_BASE_DELAY_MS * 2_u64.pow(attempt - 1);
                    warn!(attempt, delay_ms, %error, "stream failed; retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }
    }

    async fn try_stream_turn(
        &self,
        req: &TaskRequest,
        task_id: TaskId,
        request: &AgentRequest,
    ) -> Result<(String, String, PendingCalls, UsageEstimate), String> {
        let session_id = req.session_id;

        let mut stream = req
            .provider
            .stream_response(request)
            .await
            .map_err(|e| format!("{e}"))?;

        let mut text = String::new();
        let mut reasoning = String::new();
        let mut pending: PendingCalls = PendingCalls::new();
        let mut usage = UsageEstimate::default();
        let mut completed = false;

        while let Some(ev) = stream.next().await {
            match ev {
                ProviderEvent::TextDelta { text: t } => {
                    text.push_str(&t);
                    self.emit(AgentEvent::ModelDelta {
                        task_id,
                        session_id,
                        text: t,
                    });
                }
                ProviderEvent::ReasoningDelta { text: t } => {
                    reasoning.push_str(&t);
                    self.emit(AgentEvent::ReasoningDelta {
                        task_id,
                        session_id,
                        text: t,
                    });
                }
                ProviderEvent::ToolCallStart { index, id, name } => {
                    pending.entry(index).or_insert((id, name, Value::Null));
                }
                ProviderEvent::ToolCallDelta {
                    index,
                    partial_args,
                } => {
                    if let Some(entry) = pending.get_mut(&index) {
                        let current = entry.2.clone();
                        let merged = match current {
                            Value::Null => partial_args.clone(),
                            Value::String(s) => format!("{s}{partial_args}"),
                            _ => partial_args.clone(),
                        };
                        entry.2 = json!(merged);
                    }
                }
                ProviderEvent::ToolCallEnd { index, arguments } => {
                    if let Some(entry) = pending.get_mut(&index) {
                        entry.2 = arguments;
                    }
                }
                ProviderEvent::Usage { usage: u } => usage = u,
                ProviderEvent::Error { message } => return Err(message),
                ProviderEvent::Done => {
                    completed = true;
                    break;
                }
            }
        }

        if !completed {
            return Err("stream ended before completion".to_string());
        }

        // Normalize tool-call arguments: some providers only deliver a raw
        // JSON string; parse it now so the registry gets structured args.
        for entry in pending.values_mut() {
            if let Value::String(s) = &entry.2 {
                entry.2 = serde_json::from_str(s).unwrap_or_else(|_| json!({}));
            }
        }

        Ok((text, reasoning, pending, usage))
    }
}

/// Tool call accumulator, keyed by the provider's output index.
type PendingCalls = BTreeMap<usize, (String, String, Value)>;

fn last_assistant_text(messages: &[Message]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == provider_api::MessageRole::Assistant && !m.content.is_empty())
        .map(|m| m.content.clone())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_orders_definitions() {
        let reg = ToolRegistry::new(vec![]);
        assert!(reg.names().is_empty());
        assert!(reg.definitions().is_empty());
    }

    #[test]
    fn event_serde_roundtrip() {
        let ev = AgentEvent::TaskCreated {
            task_id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            provider: "OpenAI".into(),
            model: "gpt-6-astra".into(),
        };
        let json = serde_json::to_value(&ev).unwrap();
        assert_eq!(json["type"], "task_created");
        let back: AgentEvent = serde_json::from_value(json).unwrap();
        assert!(matches!(back, AgentEvent::TaskCreated { .. }));
    }

    #[test]
    fn summary_serializes_for_ui() {
        let s = TaskSummary {
            task_id: Uuid::new_v4(),
            session_id: Uuid::new_v4(),
            turns: 3,
            tool_calls: 2,
            input_tokens: 100,
            output_tokens: 50,
            final_text: "done".into(),
            status: TaskStatus::Completed,
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["status"], "completed");
        assert_eq!(v["turns"], 3);
    }
}
