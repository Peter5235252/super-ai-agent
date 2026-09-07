//! OpenAI-compatible provider adapter (`/chat/completions` + SSE).
//!
//! One wire format covers Mistral (La Plateforme), Google Gemini (via its
//! OpenAI-compat shim), Ollama, LM Studio, vLLM and any other
//! OpenAI-compatible server: same request shape, only base URL, key and
//! model id change. Local servers need no key; cloud ones do.
//!
//! The mapper is deliberately lenient: some servers report tool calls with
//! `finish_reason: "stop"`, send full `message.tool_calls` in stream
//! chunks, or close the stream without `[DONE]`. All of those still yield
//! proper [`ProviderEvent::ToolCallStart`] / `ToolCallEnd` sequences.
#![forbid(unsafe_code)]

pub mod models;

use std::collections::BTreeMap;
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

pub const MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1";
pub const GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai";
pub const OLLAMA_BASE_URL: &str = "http://localhost:11434/v1";
pub const LMSTUDIO_BASE_URL: &str = "http://localhost:1234/v1";

/// Generic client for OpenAI-style `/chat/completions` endpoints.
pub struct CompatProvider {
    http: Client,
    api_key: Option<SecretString>,
    base_url: String,
    default_model: String,
    kind: ProviderKind,
    known: fn() -> Vec<ModelInfo>,
}

impl CompatProvider {
    pub fn new(
        api_key: Option<SecretString>,
        base_url: Option<String>,
        default_model: Option<String>,
        fallback_base_url: &str,
        fallback_model: &str,
        kind: ProviderKind,
        known: fn() -> Vec<ModelInfo>,
    ) -> Self {
        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client build cannot fail"),
            api_key,
            base_url: base_url.unwrap_or_else(|| fallback_base_url.to_string()),
            default_model: default_model.unwrap_or_else(|| fallback_model.to_string()),
            kind,
            known,
        }
    }

    /// Mistral La Plateforme (`https://api.mistral.ai/v1`). Key required.
    pub fn mistral(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self::new(
            Some(api_key),
            base_url,
            default_model,
            MISTRAL_BASE_URL,
            "mistral-small-latest",
            ProviderKind::Mistral,
            models::known_models_mistral,
        )
    }

    /// Google Gemini via the OpenAI-compat shim. Key required (AI Studio).
    pub fn gemini(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self::new(
            Some(api_key),
            base_url,
            default_model,
            GEMINI_BASE_URL,
            "gemini-3.8-flash",
            ProviderKind::Google,
            models::known_models_gemini,
        )
    }

    /// Ollama (`http://localhost:11434/v1`). No key needed.
    pub fn ollama(base_url: Option<String>, default_model: Option<String>) -> Self {
        Self::new(
            None,
            base_url,
            default_model,
            OLLAMA_BASE_URL,
            "qwen3:8b",
            ProviderKind::Ollama,
            models::known_models_ollama,
        )
    }

    /// Generic local server (LM Studio default). Key optional.
    pub fn local(
        api_key: Option<SecretString>,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self::new(
            api_key,
            base_url,
            default_model,
            LMSTUDIO_BASE_URL,
            "local-model",
            ProviderKind::Local,
            models::known_models_local,
        )
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.api_key {
            Some(key) => req.bearer_auth(key.expose_secret()),
            None => req,
        }
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let url = format!("{}/models", self.base_url.trim_end_matches('/'));
        let resp = self.apply_auth(self.http.get(&url)).send().await?;
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
                                display_name: id.clone(),
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
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let model = if request.model.is_empty() {
            self.default_model.as_str()
        } else {
            request.model.as_str()
        };
        let mut body = json!({
            "model": model,
            "messages": build_messages(&request.messages, request.system.as_deref()),
            "stream": true,
        });
        if !request.tools.is_empty() {
            let tools: Vec<Value> = request
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
        }
        if let Some(max) = request.max_tokens {
            body["max_tokens"] = json!(max);
        }

        let resp = self
            .apply_auth(self.http.post(&url))
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            return Err(http_error(resp).await);
        }

        let sse = parse_sse(resp)?;
        let mapped = futures::stream::unfold(
            (sse, StreamAcc::default()),
            |(mut sse, mut acc)| async move {
                loop {
                    match sse.next().await {
                        Some(ev) => {
                            let out = map_chat_chunk(&mut acc, &ev.data);
                            if out.is_empty() {
                                continue;
                            }
                            return Some((out, (sse, acc)));
                        }
                        None => {
                            if acc.finished {
                                return None;
                            }
                            acc.finished = true;
                            return Some((flush_acc(&mut acc), (sse, acc)));
                        }
                    }
                }
            },
        )
        .flat_map(futures::stream::iter);
        Ok(Box::pin(mapped))
    }
}

#[async_trait]
impl ModelProvider for CompatProvider {
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

fn build_messages(messages: &[Message], system: Option<&str>) -> Vec<Value> {
    let mut items = Vec::new();
    if let Some(system) = system.filter(|s| !s.is_empty()) {
        items.push(json!({"role": "system", "content": system}));
    }
    for m in messages {
        match m.role {
            MessageRole::System => items.push(json!({"role": "system", "content": m.content})),
            MessageRole::User => items.push(json!({"role": "user", "content": m.content})),
            MessageRole::Assistant => {
                let mut item = json!({"role": "assistant", "content": m.content});
                if !m.tool_calls.is_empty() {
                    let calls: Vec<Value> = m
                        .tool_calls
                        .iter()
                        .map(|tc| {
                            let args = serde_json::to_string(&tc.arguments)
                                .unwrap_or_else(|_| "{}".into());
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": args},
                            })
                        })
                        .collect();
                    item["tool_calls"] = Value::Array(calls);
                }
                items.push(item);
            }
            MessageRole::Tool => items.push(json!({
                "role": "tool",
                "tool_call_id": m.tool_call_id.as_deref().unwrap_or(""),
                "content": m.content,
            })),
        }
    }
    items
}

// ---------------------------------------------------------------------------
// Lenient chat-completions SSE mapping
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct CallAcc {
    id: String,
    name: String,
    args: String,
    started: bool,
    ended: bool,
}

#[derive(Debug, Default)]
struct StreamAcc {
    calls: BTreeMap<usize, CallAcc>,
    finished: bool,
}

/// Map one SSE `data` payload (or `[DONE]`) to provider events.
fn map_chat_chunk(acc: &mut StreamAcc, data: &str) -> Vec<ProviderEvent> {
    if acc.finished {
        return vec![];
    }
    if data.trim() == "[DONE]" {
        acc.finished = true;
        return flush_acc(acc);
    }
    let chunk: Value = match serde_json::from_str(data) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    if let Some(message) = chunk
        .get("error")
        .and_then(|e| e.get("message").and_then(Value::as_str))
    {
        return vec![ProviderEvent::Error {
            message: message.to_string(),
        }];
    }

    let mut out = Vec::new();
    let choice = &chunk["choices"][0];

    // Streaming delta: text and/or tool-call fragments.
    let delta = &choice["delta"];
    if let Some(text) = delta.get("content").and_then(Value::as_str)
        && !text.is_empty()
    {
        out.push(ProviderEvent::TextDelta {
            text: text.to_string(),
        });
    }
    if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
        for tc in calls {
            let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
            let entry = acc.calls.entry(index).or_default();
            if let Some(id) = tc.get("id").and_then(Value::as_str)
                && !id.is_empty()
            {
                entry.id = id.to_string();
            }
            if let Some(name) = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                && !name.is_empty()
            {
                entry.name = name.to_string();
            }
            if !entry.started && (!entry.id.is_empty() || !entry.name.is_empty()) {
                entry.started = true;
                out.push(ProviderEvent::ToolCallStart {
                    index,
                    id: entry.id.clone(),
                    name: entry.name.clone(),
                });
            }
            if let Some(frag) = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                && !frag.is_empty()
            {
                entry.args.push_str(frag);
                out.push(ProviderEvent::ToolCallDelta {
                    index,
                    partial_args: frag.to_string(),
                });
            }
        }
    }

    // Non-stream form: a complete message (with tool calls) inside a chunk.
    // Some compat servers send this instead of deltas, sometimes even with
    // `finish_reason: "stop"`.
    let message = &choice["message"];
    if message.is_object() {
        if let Some(text) = message.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            out.push(ProviderEvent::TextDelta {
                text: text.to_string(),
            });
        }
        if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
            for (pos, tc) in calls.iter().enumerate() {
                let index = tc
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(pos as u64) as usize;
                let entry = acc.calls.entry(index).or_default();
                if let Some(id) = tc.get("id").and_then(Value::as_str) {
                    entry.id = id.to_string();
                }
                if let Some(name) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                {
                    entry.name = name.to_string();
                }
                if !entry.started {
                    entry.started = true;
                    out.push(ProviderEvent::ToolCallStart {
                        index,
                        id: entry.id.clone(),
                        name: entry.name.clone(),
                    });
                }
                let args = tc
                    .get("function")
                    .and_then(|f| f.get("arguments"))
                    .map(|a| match a {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                entry.args = args;
                entry.ended = true;
                out.push(ProviderEvent::ToolCallEnd {
                    index,
                    arguments: serde_json::from_str(&entry.args).unwrap_or_else(|_| json!({})),
                });
            }
        }
    }

    // Usage accounting when the server reports it.
    if let Some(usage) = parse_usage(&chunk) {
        out.push(ProviderEvent::Usage { usage });
    }

    // Turn completion: close out any calls the server considers finished.
    match choice.get("finish_reason").and_then(Value::as_str) {
        Some("tool_calls") | Some("stop") | Some("length") => {
            for (index, entry) in acc.calls.iter_mut() {
                if entry.started && !entry.ended {
                    entry.ended = true;
                    out.push(ProviderEvent::ToolCallEnd {
                        index: *index,
                        arguments: serde_json::from_str(&entry.args).unwrap_or_else(|_| json!({})),
                    });
                }
            }
        }
        _ => {}
    }

    out
}

fn parse_usage(chunk: &Value) -> Option<UsageEstimate> {
    let usage = chunk.get("usage")?;
    let input = usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if input == 0 && output == 0 {
        return None;
    }
    Some(UsageEstimate {
        input_tokens: input,
        output_tokens: output,
        cost_usd: None,
    })
}

/// Flush pending calls and terminate the stream. Idempotent enough for the
/// unfold driver: it runs once when the body ends or `[DONE]` arrives.
fn flush_acc(acc: &mut StreamAcc) -> Vec<ProviderEvent> {
    let mut out = Vec::new();
    for (index, entry) in acc.calls.iter_mut() {
        if entry.started && !entry.ended {
            entry.ended = true;
            out.push(ProviderEvent::ToolCallEnd {
                index: *index,
                arguments: serde_json::from_str(&entry.args).unwrap_or_else(|_| json!({})),
            });
        }
    }
    out.push(ProviderEvent::Done);
    out
}

#[allow(dead_code)]
fn sse_event(data: &str) -> SseEvent {
    SseEvent {
        event: None,
        data: data.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use provider_api::ToolCall;

    #[test]
    fn messages_convert_all_roles() {
        let messages = vec![
            Message::system("sys"),
            Message::user("hi"),
            Message::assistant(
                "checking",
                vec![ToolCall::new("call_1", "fs.read", json!({"path": "x"}))],
            ),
            Message::tool("call_1", "contents"),
        ];
        let items = build_messages(&messages, Some("global"));
        assert_eq!(items.len(), 5);
        assert_eq!(items[0]["role"], "system");
        assert_eq!(items[0]["content"], "global");
        assert_eq!(items[3]["role"], "assistant");
        assert_eq!(items[3]["tool_calls"][0]["id"], "call_1");
        assert_eq!(items[3]["tool_calls"][0]["type"], "function");
        assert_eq!(items[4]["role"], "tool");
        assert_eq!(items[4]["tool_call_id"], "call_1");
    }

    #[test]
    fn text_delta_maps() {
        let mut acc = StreamAcc::default();
        let out = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#,
        );
        assert_eq!(out.len(), 1);
        assert!(matches!(
            &out[0],
            ProviderEvent::TextDelta { text } if text == "hello"
        ));
    }

    #[test]
    fn streamed_tool_call_assembles() {
        let mut acc = StreamAcc::default();
        let start = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"fs.read","arguments":""}}]},"finish_reason":null}]}"#,
        );
        assert!(matches!(
            &start[0],
            ProviderEvent::ToolCallStart { id, name, .. } if id == "call_1" && name == "fs.read"
        ));
        // Argument fragments are built with json! so no hand-escaping can
        // corrupt them; together they form {"path":"x"}.
        let frag1 = "{\"pa";
        let frag2 = "th\":\"x\"}";
        let d1 = serde_json::json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": frag1}}]}, "finish_reason": null}]}).to_string();
        let d1 = map_chat_chunk(&mut acc, &d1);
        assert!(matches!(&d1[0], ProviderEvent::ToolCallDelta { .. }));
        let d2 = serde_json::json!({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": frag2}}]}, "finish_reason": null}]}).to_string();
        let d2 = map_chat_chunk(&mut acc, &d2);
        assert!(matches!(&d2[0], ProviderEvent::ToolCallDelta { .. }));
        let end = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#,
        );
        assert!(matches!(
            &end[0],
            ProviderEvent::ToolCallEnd { arguments, .. } if arguments["path"] == "x"
        ));
    }

    #[test]
    fn non_stream_tool_call_with_stop_finish_maps() {
        // Gemini-compat quirk: full tool call + finish_reason "stop".
        let mut acc = StreamAcc::default();
        let out = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"message":{"role":"assistant","content":"","tool_calls":[{"id":"call_9","type":"function","function":{"name":"process.run","arguments":"{\"command\":\"dir\"}"}}]},"finish_reason":"stop"}]}"#,
        );
        assert!(matches!(&out[0], ProviderEvent::ToolCallStart { .. }));
        assert!(matches!(
            &out[1],
            ProviderEvent::ToolCallEnd { arguments, .. } if arguments["command"] == "dir"
        ));
    }

    #[test]
    fn done_flushes_pending_call() {
        let mut acc = StreamAcc::default();
        let _ = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"fs.list","arguments":""}}]},"finish_reason":null}]}"#,
        );
        let out = map_chat_chunk(&mut acc, "[DONE]");
        assert!(matches!(&out[0], ProviderEvent::ToolCallEnd { .. }));
        assert!(matches!(&out[1], ProviderEvent::Done));
    }

    #[test]
    fn error_chunk_maps() {
        let mut acc = StreamAcc::default();
        let out = map_chat_chunk(&mut acc, r#"{"error":{"message":"boom"}}"#);
        assert!(matches!(&out[0], ProviderEvent::Error { .. }));
        let out2 = map_chat_chunk(&mut acc, "not json at all");
        assert!(out2.is_empty());
    }

    #[test]
    fn usage_chunk_maps() {
        let mut acc = StreamAcc::default();
        let out = map_chat_chunk(
            &mut acc,
            r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#,
        );
        assert!(out.iter().any(|e| matches!(
            e,
            ProviderEvent::Usage { usage } if usage.input_tokens == 10 && usage.output_tokens == 5
        )));
    }

    #[test]
    fn catalogs_cover_new_families() {
        let mistral: Vec<String> = models::known_models_mistral()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert!(mistral.contains(&"mistral-large-latest".to_string()));
        assert!(mistral.contains(&"codestral-latest".to_string()));
        let gemini: Vec<String> = models::known_models_gemini()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert!(gemini.contains(&"gemini-3.8-flash".to_string()));
        assert!(gemini.contains(&"gemini-3.1-pro-preview".to_string()));
        let ollama: Vec<String> = models::known_models_ollama()
            .into_iter()
            .map(|m| m.id)
            .collect();
        assert!(ollama.contains(&"qwen3:8b".to_string()));
        assert_eq!(models::resolve_local("my-gguf").id, "my-gguf");
        assert_eq!(models::resolve_mistral("future-x").id, "future-x");
    }
}
