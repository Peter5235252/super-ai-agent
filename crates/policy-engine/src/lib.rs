//! Capability-based policy engine.
//!
//! Every tool operation is checked before execution:
//!
//! ```text
//! required capabilities + target resource + risk score + current policy
//! ```
//!
//! The default policy auto-approves low-risk operations and routes
//! medium+ risk through a human approval gate. Hard denials (credentials,
//! secrets) never reach the human — they are simply refused.
#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;

use app_core::{ApprovalDecision, RiskClass};
use serde_json::Value;
use tokio::sync::{Mutex, oneshot};
use tool_core::{ToolContext, ToolDefinition};
use tracing::warn;
use uuid::Uuid;

/// Result of a policy evaluation.
#[derive(Debug)]
pub enum Decision {
    Allow,
    Deny {
        reason: String,
    },
    /// Needs a human; `receiver` resolves when they answer.
    PendingApproval {
        id: Uuid,
        reason: String,
        receiver: oneshot::Receiver<ApprovalDecision>,
    },
}

#[derive(Debug)]
pub struct PolicyEngine {
    auto_approve: Mutex<RiskClass>,
    pending: Mutex<HashMap<Uuid, oneshot::Sender<ApprovalDecision>>>,
}

impl PolicyEngine {
    pub fn new(auto_approve: RiskClass) -> Arc<Self> {
        Arc::new(Self {
            auto_approve: Mutex::new(auto_approve),
            pending: Mutex::new(HashMap::new()),
        })
    }

    /// Autonomy knob: operations at or below this risk run automatically.
    pub async fn set_auto_approve(&self, risk: RiskClass) {
        *self.auto_approve.lock().await = risk;
    }

    /// Evaluate a tool call. Never blocks; `PendingApproval` carries a
    /// receiver the caller awaits after notifying the UI.
    pub async fn authorize(
        &self,
        def: &ToolDefinition,
        _args: &Value,
        ctx: &ToolContext,
    ) -> Decision {
        // Hard denials: capabilities the model should never reach, period.
        if def.name.starts_with("credential.") || def.name.starts_with("secret.") {
            return Decision::Deny {
                reason: format!(
                    "tool '{}' accesses credentials and is denied by policy",
                    def.name
                ),
            };
        }
        if def.network && false {
            // reserved: network policy enforcement arrives with the sandbox phase
        }

        let mut risk = def.risk;
        let mut reason = String::new();
        if def.workspace_scoped && ctx.workspace.is_none() {
            risk = risk.escalate();
            reason =
                "workspace-scoped tool invoked without a workspace; risk escalated".to_string();
        }

        let threshold = *self.auto_approve.lock().await;
        if risk <= threshold {
            return Decision::Allow;
        }

        let (tx, rx) = oneshot::channel();
        let id = Uuid::new_v4();
        self.pending.lock().await.insert(id, tx);
        let reason = if reason.is_empty() {
            format!("{} risk operation", risk.label())
        } else {
            reason
        };
        Decision::PendingApproval {
            id,
            reason,
            receiver: rx,
        }
    }

    /// Submit a human's answer. Returns false if the request already
    /// timed out or was answered.
    pub async fn respond(&self, id: Uuid, decision: ApprovalDecision) -> bool {
        let mut pending = self.pending.lock().await;
        match pending.remove(&id) {
            Some(tx) => {
                let _ = tx.send(decision);
                true
            }
            None => {
                warn!(%id, "approval response for unknown request");
                false
            }
        }
    }

    pub async fn pending_count(&self) -> usize {
        self.pending.lock().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::{SideEffect, WorkspaceConfig};
    use tool_core::ToolDefinition;

    fn def(name: &str, risk: RiskClass, scoped: bool) -> ToolDefinition {
        let mut d = ToolDefinition::new(
            name,
            "test",
            Value::Null,
            risk,
            vec![SideEffect::ReadsFiles],
        );
        d.workspace_scoped = scoped;
        d
    }

    #[tokio::test]
    async fn low_risk_auto_approved() {
        let p = PolicyEngine::new(RiskClass::Low);
        let d = def("fs.read", RiskClass::Low, false);
        assert!(matches!(
            p.authorize(&d, &Value::Null, &ToolContext::default()).await,
            Decision::Allow
        ));
    }

    #[tokio::test]
    async fn medium_risk_requires_approval_flow() {
        let p = PolicyEngine::new(RiskClass::Low);
        let d = def("fs.write", RiskClass::Medium, false);
        match p.authorize(&d, &Value::Null, &ToolContext::default()).await {
            Decision::PendingApproval { id, receiver, .. } => {
                assert!(p.respond(id, ApprovalDecision::AllowOnce).await);
                assert_eq!(receiver.await.unwrap(), ApprovalDecision::AllowOnce);
            }
            other => panic!("expected pending approval, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn scoped_tool_without_workspace_escalates() {
        let p = PolicyEngine::new(RiskClass::Medium);
        // fs.read is Low but workspace-scoped; without a workspace it escalates to Medium,
        // which is at the threshold → still allowed.
        let d = def("fs.read", RiskClass::Low, true);
        assert!(matches!(
            p.authorize(&d, &Value::Null, &ToolContext::default()).await,
            Decision::Allow
        ));

        // With the threshold at Low, escalation to Medium requires approval.
        let p2 = PolicyEngine::new(RiskClass::Low);
        assert!(matches!(
            p2.authorize(&d, &Value::Null, &ToolContext::default())
                .await,
            Decision::PendingApproval { .. }
        ));
    }

    #[tokio::test]
    async fn credentials_hard_denied() {
        let p = PolicyEngine::new(RiskClass::Critical);
        let d = def("credential.read", RiskClass::Low, false);
        assert!(matches!(
            p.authorize(&d, &Value::Null, &ToolContext::default()).await,
            Decision::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn deny_answer_reaches_agent() {
        let p = PolicyEngine::new(RiskClass::Low);
        let d = def("fs.write", RiskClass::Medium, false);
        match p.authorize(&d, &Value::Null, &ToolContext::default()).await {
            Decision::PendingApproval { id, receiver, .. } => {
                assert!(p.respond(id, ApprovalDecision::Deny).await);
                assert_eq!(receiver.await.unwrap(), ApprovalDecision::Deny);
            }
            other => panic!("expected pending approval, got {other:?}"),
        }
        let _ = WorkspaceConfig::from_dir("x", std::path::PathBuf::from("/tmp"));
    }
}
