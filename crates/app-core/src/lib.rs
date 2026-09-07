//! Shared domain types for Super-AI.
//!
//! This is the lowest layer of the dependency graph. Nearly every other crate
//! depends on it, so it stays small and dependency-light.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Identifiers are plain `Uuid`s for now; typed newtypes can be introduced
/// later without changing the storage format.
pub type SessionId = Uuid;
pub type TaskId = Uuid;

/// Risk classification used by the policy engine and approval UI.
///
/// Ordered: `Low < Medium < High < Critical`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskClass {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskClass {
    pub fn label(self) -> &'static str {
        match self {
            RiskClass::Low => "LOW",
            RiskClass::Medium => "MEDIUM",
            RiskClass::High => "HIGH",
            RiskClass::Critical => "CRITICAL",
        }
    }

    /// Escalate one level. Used when an operation leaves its declared scope
    /// (e.g. a workspace-scoped tool invoked without a workspace).
    pub fn escalate(self) -> RiskClass {
        match self {
            RiskClass::Low => RiskClass::Medium,
            RiskClass::Medium => RiskClass::High,
            RiskClass::High => RiskClass::Critical,
            RiskClass::Critical => RiskClass::Critical,
        }
    }
}

/// What a human decided about an approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    AllowOnce,
    AllowForTask,
    AllowForWorkspace,
    AlwaysAllow,
    Deny,
}

impl ApprovalDecision {
    pub fn is_deny(self) -> bool {
        matches!(self, ApprovalDecision::Deny)
    }
}

/// Declared side effects of a tool, shown to the user before approval and
/// stored in the audit log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SideEffect {
    ReadsFiles,
    WritesFiles,
    DeletesFiles,
    RunsCommands,
    NetworkRequest,
    InstallsSoftware,
    ModifiesSystem,
    ControlsGui,
    AccessesCredentials,
    AccessesClipboard,
}

/// A workspace is more than a folder: it carries filesystem roots, rules,
/// and (in later phases) memory, MCP servers, permissions and task history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub id: Uuid,
    pub name: String,
    pub roots: Vec<PathBuf>,
    pub created_at_ms: i64,
}

impl WorkspaceConfig {
    pub fn from_dir(name: impl Into<String>, dir: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            roots: vec![dir],
            created_at_ms: now_ms(),
        }
    }

    /// Primary root, if any.
    pub fn root(&self) -> Option<&PathBuf> {
        self.roots.first()
    }
}

/// Application-level settings (stored as key/value rows in SQLite).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    /// Risk threshold that is approved automatically (no human prompt).
    pub auto_approve_risk: RiskClass,
    pub max_task_turns: u32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            default_provider: None,
            default_model: None,
            auto_approve_risk: RiskClass::Low,
            max_task_turns: 32,
        }
    }
}

/// Current unix time in milliseconds (used for all persisted timestamps).
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
