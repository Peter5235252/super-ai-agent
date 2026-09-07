//! Tool abstraction: every capability of the agent is a typed, self-
//! describing tool with an explicit risk class, side effects and scope.
//! The model only ever sees the `name`/`description`/`input_schema` wire
//! form; policy and approval decisions use the full definition.
#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

pub use app_core::{ApprovalDecision, RiskClass, SideEffect, WorkspaceConfig};

#[derive(Debug, Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing `execute` arguments.
    pub input_schema: Value,
    /// Base risk class, escalated by the policy engine when the operation
    /// leaves its declared scope.
    pub risk: RiskClass,
    pub side_effects: Vec<SideEffect>,
    pub timeout: Duration,
    /// If true, the tool is only meaningful inside a workspace root.
    pub workspace_scoped: bool,
    /// Whether the tool makes outbound network requests.
    pub network: bool,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        risk: RiskClass,
        side_effects: Vec<SideEffect>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
            risk,
            side_effects,
            timeout: Duration::from_secs(30),
            workspace_scoped: false,
            network: false,
        }
    }
}

/// Execution context passed to every tool call.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    pub workspace: Option<WorkspaceConfig>,
    pub cwd: Option<PathBuf>,
}

impl ToolContext {
    pub fn root(&self) -> Option<&PathBuf> {
        self.workspace.as_ref().and_then(|w| w.root())
    }
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub output: String,
    pub truncated: bool,
}

impl ToolResult {
    pub fn text(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            truncated: false,
        }
    }

    pub fn truncated(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            truncated: true,
        }
    }
}

impl std::fmt::Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.output)
    }
}

/// A typed capability. Implementations must be stateless or internally
/// synchronized; the agent may call them concurrently.
#[async_trait]
pub trait Tool: Send + Sync {
    fn definition(&self) -> &ToolDefinition;
    async fn execute(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolResult>;
}
