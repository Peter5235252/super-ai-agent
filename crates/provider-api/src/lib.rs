//! Provider-neutral AI interface.
//!
//! The agent runtime talks only to [`ModelProvider`]. Concrete adapters
//! (OpenAI, Anthropic, xAI, Google, ...) live in separate crates and are
//! registered by the application shell, so the agent brain never depends on
//! any particular vendor.
#![forbid(unsafe_code)]

pub mod sse;

use async_trait::async_trait;
use futures::Stream;
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use thiserror::Error;

/// Vendors / protocol families we can talk to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    OpenAI,
    Anthropic,
    Google,
    Xai,
    Mistral,
    Ollama,
    Local,
    OpenAiCompatible,
    Custom,
}

impl ProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            ProviderKind::OpenAI => "OpenAI",
            ProviderKind::Anthropic => "Anthropic",
            ProviderKind::Google => "Google",
            ProviderKind::Xai => "SpaceXAI (Grok)",
            ProviderKind::Mistral => "Mistral",
            ProviderKind::Ollama => "Ollama",
            ProviderKind::Local => "Local",
            ProviderKind::OpenAiCompatible => "OpenAI-compatible",
            ProviderKind::Custom => "Custom",
        }
    }
}

/// Explicit model capabilities. The runtime uses these to pick execution
/// strategies (e.g. native web search vs. the app's own web tool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Capability {
    TextGeneration,
    Vision,
    Audio,
    ToolCalling,
    WebSearch,
    ComputerUse,
    Reasoning,
    StructuredOutput,
    Streaming,
    LongContext,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub text_generation: bool,
    pub vision: bool,
    pub audio: bool,
    pub tool_calling: bool,
    pub web_search: bool,
    pub computer_use: bool,
    pub reasoning: bool,
    pub structured_output: bool,
    pub streaming: bool,
    pub long_context: bool,
}

impl CapabilitySet {
    pub fn has(&self, cap: Capability) -> bool {
        match cap {
            Capability::TextGeneration => self.text_generation,
            Capability::Vision => self.vision,
            Capability::Audio => self.audio,
            Capability::ToolCalling => self.tool_calling,
            Capability::WebSearch => self.web_search,
            Capability::ComputerUse => self.computer_use,
            Capability::Reasoning => self.reasoning,
            Capability::StructuredOutput => self.structured_output,
            Capability::Streaming => self.streaming,
            Capability::LongContext => self.long_context,
        }
    }

    /// Sensible defaults for a modern chat-capable model.
    pub fn chat_default() -> Self {
        Self {
            text_generation: true,
            vision: false,
            audio: false,
            tool_calling: true,
            web_search: false,
            computer_use: false,
            reasoning: true,
            structured_output: true,
            streaming: true,
            long_context: true,
        }
    }
}

/// A model the user can select, with metadata for routing and cost display.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Exact model id used in API requests (e.g. `gpt-5.6-sol`).
    pub id: String,
    pub provider: ProviderKind,
    pub display_name: String,
    pub capabilities: CapabilitySet,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    /// USD per 1M input tokens (approximate where marked in `notes`).
    pub input_price_per_mtok: Option<f64>,
    pub output_price_per_mtok: Option<f64>,
    pub knowledge_cutoff: Option<String>,
    pub notes: Option<String>,
}

/// Canonical, vendor-neutral message representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed JSON arguments.
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }
    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls,
            tool_call_id: None,
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: vec![],
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

/// Tool definition in the shape providers need (name/description/schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinitionWire {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// One model-call request. Providers convert this into their native format.
#[derive(Debug, Clone)]
pub struct AgentRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinitionWire>,
    pub system: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Token + optional cost accounting for a completed model call.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct UsageEstimate {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: Option<f64>,
}

/// Streaming events emitted by a provider while generating a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProviderEvent {
    TextDelta {
        text: String,
    },
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallDelta {
        index: usize,
        partial_args: String,
    },
    ToolCallEnd {
        index: usize,
        arguments: serde_json::Value,
    },
    Usage {
        usage: UsageEstimate,
    },
    Done,
    Error {
        message: String,
    },
}

pub type ProviderEventStream = Pin<Box<dyn Stream<Item = ProviderEvent> + Send>>;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("http {status}: {body}")]
    Http { status: u16, body: String },
    #[error("authentication failed: {0}")]
    Auth(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("streaming error: {0}")]
    Streaming(String),
    #[error("request timed out")]
    Timeout,
}

pub type Result<T> = std::result::Result<T, ProviderError>;

/// The vendor-neutral model interface.
#[async_trait]
pub trait ModelProvider: Send + Sync {
    fn kind(&self) -> ProviderKind;
    fn default_capabilities(&self) -> CapabilitySet {
        CapabilitySet::chat_default()
    }
    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream>;
}

/// Convenience: construct a provider from a stored key. Adapter crates
/// implement this for their own types.
pub trait ProviderFactory {
    type Provider: ModelProvider + 'static;
    fn from_secret(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self::Provider;
}
