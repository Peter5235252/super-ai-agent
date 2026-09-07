//! OpenAI provider adapter (Responses API) with live SSE streaming and
//! function calling. Also the base for other Responses-API-compatible
//! providers such as xAI's Grok.
#![forbid(unsafe_code)]

pub mod models;

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use provider_api::sse::{SseEvent, http_error, parse_sse};
use provider_api::{
    AgentRequest, CapabilitySet, Message, MessageRole, ModelInfo, ModelProvider, ProviderEvent,
    ProviderEventStream, ProviderKind, Result, UsageEstimate,
};
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use serde_json::{Value, json};

pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_MODEL: &str = "gpt-6-astra";

/// Generic client for OpenAI-style `/responses` endpoints. `kind` and the
/// `known` metadata table let xAI and other compatible services reuse it.
pub struct ResponsesClient {
    http: Client,
    api_key: SecretString,
    base_url: String,
    default_model: String,
    kind: ProviderKind,
    known: fn() -> Vec<ModelInfo>,
}

impl ResponsesClient {
    pub fn new(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
        kind: ProviderKind,
        known: fn() -> Vec<ModelInfo>,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client build cannot fail"),
            api_key,
            base_url: base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string()),
            default_model: default_model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            kind,
            known,
        }
    }

    fn headers(&self) -> String {
        self.api_key.expose_secret().to_string()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .get(&url)
            .bearer_auth(self.headers())
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }
        let body: Value = resp.json().await?;
        let known = (self.known)();
        let fallback_caps = CapabilitySet::chat_default();
        Ok(body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m["id"].as_str().map(str::to_string))
                    .map(|id| {
                        known
                            .iter()
                            .find(|k| k.id == id)
                            .cloned()
                            .unwrap_or_else(|| ModelInfo {
                                id: id.clone(),
                                provider: self.kind,
                                display_name: id,
                                capabilities: fallback_caps.clone(),
                                context_window: None,
                                max_output_tokens: None,
                                input_price_per_mtok: None,
                                output_price_per_mtok: None,
                                knowledge_cutoff: None,
                                notes: Some(
                                    "Unlisted model; using generic capability defaults.".into(),
                                ),
                            })
                    })
                    .collect()
            })
            .unwrap_or_default())
    }

    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream> {
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let model = if request.model.is_empty() {
            self.default_model.as_str()
        } else {
            request.model.as_str()
        };
        let mut body = json!({
            "model": model,
            "input": build_input(&request.messages),
            "stream": true,
            "store": false,
        });
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }
        if let Some(max) = request.max_tokens {
            body["max_output_tokens"] = json!(max);
        }

        let resp = self
            .http
            .post(&url)
            .bearer_auth(self.headers())
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }

        let sse = parse_sse(resp)?;
        Ok(Box::pin(sse.flat_map(|ev| {
            futures::stream::iter(map_responses_event(ev))
        })))
    }
}

#[async_trait]
impl ModelProvider for ResponsesClient {
    fn kind(&self) -> ProviderKind {
        self.kind
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.list_models().await
    }

    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream> {
        self.stream_response(request).await
    }
}

/// Thin OpenAI-branded wrapper.
pub struct OpenAiProvider(pub ResponsesClient);

impl OpenAiProvider {
    pub fn new(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self(ResponsesClient::new(
            api_key,
            base_url,
            default_model,
            ProviderKind::OpenAI,
            models::known_models,
        ))
    }
}

#[async_trait]
impl ModelProvider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        self.0.kind()
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.0.list_models().await
    }

    async fn stream_response(&self, request: &AgentRequest) -> Result<ProviderEventStream> {
        self.0.stream_response(request).await
    }
}

fn build_input(messages: &[Message]) -> Vec<Value> {
    let mut items = Vec::new();
    for m in messages {
        match m.role {
            MessageRole::System => items.push(json!({"role": "system", "content": m.content})),
            MessageRole::User => items.push(json!({"role": "user", "content": m.content})),
            MessageRole::Assistant => {
                if !m.content.is_empty() {
                    items.push(json!({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": m.content}],
                    }));
                }
                for tc in &m.tool_calls {
                    let args = serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".into());
                    items.push(json!({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": args,
                    }));
                }
            }
            MessageRole::Tool => items.push(json!({
                "type": "function_call_output",
                "call_id": m.tool_call_id.as_deref().unwrap_or(""),
                "output": m.content,
            })),
        }
    }
    items
}

fn map_responses_event(ev: SseEvent) -> Vec<ProviderEvent> {
    let data: Value = match serde_json::from_str(&ev.data) {
        Ok(d) => d,
        Err(_) => return vec![],
    };
    let kind = match data.get("type").and_then(Value::as_str) {
        Some(k) => k,
        None => return vec![],
    };
    match kind {
        "response.output_text.delta" => data["delta"]["text"]
            .as_str()
            .map(|t| {
                vec![ProviderEvent::TextDelta {
                    text: t.to_string(),
                }]
            })
            .unwrap_or_default(),
        "response.output_item.added" => match data["item"]["type"].as_str() {
            Some("function_call") => {
                let index = data["output_index"].as_u64().unwrap_or(0) as usize;
                let id = data["item"]["call_id"].as_str().unwrap_or("").to_string();
                let name = data["item"]["name"].as_str().unwrap_or("").to_string();
                vec![ProviderEvent::ToolCallStart { index, id, name }]
            }
            _ => vec![],
        },
        "response.function_call_arguments.delta" => {
            let index = data["output_index"].as_u64().unwrap_or(0) as usize;
            data["delta"]
                .as_str()
                .map(|d| {
                    vec![ProviderEvent::ToolCallDelta {
                        index,
                        partial_args: d.to_string(),
                    }]
                })
                .unwrap_or_default()
        }
        "response.output_item.done" => {
            if data["item"]["type"].as_str() == Some("function_call") {
                let index = data["output_index"].as_u64().unwrap_or(0) as usize;
                let args = data["item"]["arguments"]
                    .as_str()
                    .and_then(|s| serde_json::from_str(s).ok())
                    .unwrap_or_else(|| json!({}));
                vec![ProviderEvent::ToolCallEnd {
                    index,
                    arguments: args,
                }]
            } else {
                vec![]
            }
        }
        "response.completed" => {
            let usage = &data["response"]["usage"];
            let usage = UsageEstimate {
                input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
                output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
                cost_usd: None,
            };
            vec![ProviderEvent::Usage { usage }, ProviderEvent::Done]
        }
        "response.failed" => {
            let message = data["response"]["error"]["message"]
                .as_str()
                .or_else(|| data["error"]["message"].as_str())
                .unwrap_or("model request failed")
                .to_string();
            vec![ProviderEvent::Error { message }]
        }
        "error" => {
            let message = data["error"]["message"]
                .as_str()
                .or_else(|| data["message"].as_str())
                .unwrap_or("unknown error")
                .to_string();
            vec![ProviderEvent::Error { message }]
        }
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use provider_api::{ToolCall, ToolDefinitionWire};

    #[test]
    fn input_conversion_roundtrip() {
        let messages = vec![
            Message::system("be careful"),
            Message::user("hello"),
            Message::assistant(
                "let me check",
                vec![ToolCall::new("call_1", "fs.read", json!({"path": "x"}))],
            ),
            Message::tool("call_1", "file contents"),
        ];
        let items = build_input(&messages);
        assert_eq!(items.len(), 5);
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[1]["role"], "user");
        assert_eq!(items[2]["type"], "message");
        assert_eq!(items[3]["type"], "function_call");
        assert_eq!(items[3]["call_id"], "call_1");
        assert_eq!(items[4]["type"], "function_call_output");
        assert_eq!(items[4]["call_id"], "call_1");
    }

    #[test]
    fn known_models_include_current_family() {
        let ids: Vec<String> = models::known_models().into_iter().map(|m| m.id).collect();
        assert!(ids.iter().any(|id| id == "gpt-6-astra"));
        assert!(ids.iter().any(|id| id == "gpt-5.6-sol"));
        assert!(ids.iter().any(|id| id == "gpt-5.6-luna"));
    }

    #[test]
    fn unknown_model_resolves_to_generic() {
        let m = models::resolve_model("gpt-9.9-future");
        assert_eq!(m.id, "gpt-9.9-future");
        assert!(m.capabilities.tool_calling);
    }

    #[test]
    fn tool_wire_schema() {
        let t = ToolDefinitionWire {
            name: "fs.read".into(),
            description: "read".into(),
            input_schema: json!({"type": "object"}),
        };
        assert_eq!(t.name, "fs.read");
    }
}
