//! Anthropic provider adapter (Messages API) with live SSE streaming and
//! tool use. Converts the canonical message format to Anthropic's
//! content-block format and back.
#![forbid(unsafe_code)]

pub mod models;

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use provider_api::sse::{SseEvent, http_error, parse_sse};
use provider_api::{
    AgentRequest, Message, MessageRole, ModelInfo, ModelProvider, ProviderEvent,
    ProviderEventStream, ProviderKind, Result, UsageEstimate,
};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};

pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
pub const DEFAULT_MODEL: &str = "claude-opus-5";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8192;

pub struct AnthropicProvider {
    http: Client,
    api_key: SecretString,
    base_url: String,
    default_model: String,
}

impl AnthropicProvider {
    pub fn new(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client build cannot fail"),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            default_model: default_model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", API_VERSION)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }
        let body: Value = resp.json().await?;
        Ok(body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(str::to_string))
                    .map(|id| models::resolve_model(&id))
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream> {
        let url = format!("{}/messages", self.base_url.trim_end_matches('/'));
        let (system, messages) = build_messages(&request.messages);
        let model = if request.model.is_empty() {
            self.default_model.as_str()
        } else {
            request.model.as_str()
        };

        let mut body = json!({
            "model": model,
            "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            "stream": true,
            "messages": messages,
        });
        if let Some(system) = system {
            body["system"] = json!(system);
        }
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }

        let resp = self
            .http
            .post(&url)
            .header("x-api-key", self.api_key.expose_secret())
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }

        let sse = parse_sse(resp)?;
        let stream = sse
            .scan(AnthState::default(), |state, ev| {
                futures::future::ready(Some(map_anth_event(state, ev)))
            })
            .flat_map(futures::stream::iter);
        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl ModelProvider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.list_models().await
    }

    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream> {
        self.stream_response(request).await
    }
}

#[derive(Default)]
struct AnthState {
    usage_in: u64,
    usage_out: u64,
    /// index -> (tool_use_id, name, accumulated partial JSON)
    blocks: BTreeMap<usize, (String, String, String)>,
}

fn map_anth_event(state: &mut AnthState, ev: SseEvent) -> Vec<ProviderEvent> {
    let data: Value = match serde_json::from_str(&ev.data) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let kind = match data.get("type").and_then(Value::as_str) {
        Some(k) => k,
        None => return vec![],
    };
    match kind {
        "message_start" => {
            state.usage_in = data["message"]["usage"]["input_tokens"]
                .as_u64()
                .unwrap_or(0);
            vec![]
        }
        "content_block_start" => {
            let index = data["index"].as_u64().unwrap_or(0) as usize;
            if data["content_block"]["type"].as_str() == Some("tool_use") {
                let id = data["content_block"]["id"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let name = data["content_block"]["name"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                state
                    .blocks
                    .insert(index, (id.clone(), name.clone(), String::new()));
                vec![ProviderEvent::ToolCallStart { index, id, name }]
            } else {
                vec![]
            }
        }
        "content_block_delta" => {
            let index = data["index"].as_u64().unwrap_or(0) as usize;
            match data["delta"]["type"].as_str() {
                Some("text_delta") => data["delta"]["text"]
                    .as_str()
                    .map(|t| {
                        vec![ProviderEvent::TextDelta {
                            text: t.to_string(),
                        }]
                    })
                    .unwrap_or_default(),
                Some("input_json_delta") => {
                    if let (Some(entry), Some(partial)) = (
                        state.blocks.get_mut(&index),
                        data["delta"]["partial_json"].as_str(),
                    ) {
                        entry.2.push_str(partial);
                    }
                    vec![]
                }
                _ => vec![],
            }
        }
        "content_block_stop" => {
            let index = data["index"].as_u64().unwrap_or(0) as usize;
            if let Some((_id, _name, args)) = state.blocks.remove(&index) {
                let arguments = serde_json::from_str(&args).unwrap_or_else(|_| json!({}));
                vec![ProviderEvent::ToolCallEnd { index, arguments }]
            } else {
                vec![]
            }
        }
        "message_delta" => {
            state.usage_out = data["usage"]["output_tokens"].as_u64().unwrap_or(0);
            vec![]
        }
        "message_stop" => {
            let usage = UsageEstimate {
                input_tokens: state.usage_in,
                output_tokens: state.usage_out,
                cost_usd: None,
            };
            vec![ProviderEvent::Usage { usage }, ProviderEvent::Done]
        }
        "error" => {
            let message = data["error"]["message"]
                .as_str()
                .unwrap_or("unknown error")
                .to_string();
            vec![ProviderEvent::Error { message }]
        }
        _ => vec![],
    }
}

/// Convert canonical messages to Anthropic's wire format.
/// Returns (system prompt, message blocks). Consecutive user/tool turns are
/// merged into a single user message with multiple blocks, which the API
/// requires.
fn build_messages(messages: &[Message]) -> (Option<String>, Vec<Value>) {
    let mut system: Vec<String> = Vec::new();
    let mut msgs: Vec<Value> = Vec::new();

    for m in messages {
        match m.role {
            MessageRole::System => system.push(m.content.clone()),
            MessageRole::User => {
                let block = json!({"type": "text", "text": m.content});
                push_user_block(&mut msgs, block);
            }
            MessageRole::Assistant => {
                let mut blocks: Vec<Value> = Vec::new();
                if !m.content.is_empty() {
                    blocks.push(json!({"type": "text", "text": m.content}));
                }
                for tc in &m.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                msgs.push(json!({"role": "assistant", "content": blocks}));
            }
            MessageRole::Tool => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.as_deref().unwrap_or(""),
                    "content": m.content,
                });
                push_user_block(&mut msgs, block);
            }
        }
    }

    let system = if system.is_empty() {
        None
    } else {
        Some(system.join("\n\n"))
    };
    (system, msgs)
}

fn push_user_block(msgs: &mut Vec<Value>, block: Value) {
    if let Some(arr) = msgs
        .last_mut()
        .filter(|last| last["role"] == "user")
        .and_then(|last| last["content"].as_array_mut())
    {
        arr.push(block);
        return;
    }
    msgs.push(json!({"role": "user", "content": [block]}));
}

#[cfg(test)]
mod tests {
    use super::*;
    use provider_api::{ToolCall, ToolDefinitionWire};

    #[test]
    fn message_conversion_merges_user_turns() {
        let messages = vec![
            Message::system("sys"),
            Message::user("hello"),
            Message::assistant(
                "checking",
                vec![ToolCall::new("t1", "fs.read", json!({"path": "a"}))],
            ),
            Message::tool("t1", "contents"),
            Message::user("thanks"),
        ];
        let (system, msgs) = build_messages(&messages);
        assert_eq!(system.as_deref(), Some("sys"));
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"].as_array().unwrap().len(), 2); // text + tool_use
        // tool result + follow-up user text merged into one user message
        assert_eq!(msgs[2]["role"], "user");
        let blocks = msgs[2]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[1]["type"], "text");
    }

    #[test]
    fn known_models_include_current_family() {
        let ids: Vec<String> = models::known_models().into_iter().map(|m| m.id).collect();
        assert!(ids.iter().any(|id| id == "claude-fable-5-1"));
        assert!(ids.iter().any(|id| id == "claude-opus-5"));
        assert!(ids.iter().any(|id| id == "claude-sonnet-5"));
    }

    #[test]
    fn tool_wire() {
        let t = ToolDefinitionWire {
            name: "x".into(),
            description: "y".into(),
            input_schema: json!({"type": "object"}),
        };
        assert_eq!(t.name, "x");
    }
}
