//! Minimal Server-Sent Events parser shared by all streaming providers.
//!
//! Handles the common `event:` / `data:` line framing and blank-line
//! dispatch used by OpenAI (Responses API), Anthropic (Messages API) and
//! xAI. Comment lines (`: ...`) are ignored.

use futures::{Stream, StreamExt};
use reqwest::Response;
use std::pin::Pin;

use crate::{ProviderError, Result};

#[derive(Debug, Clone)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Parse the byte stream of an already-successful HTTP response into SSE
/// events. The returned stream ends when the body ends or decoding fails.
pub fn parse_sse(response: Response) -> Result<Pin<Box<dyn Stream<Item = SseEvent> + Send>>> {
    let byte_stream = response.bytes_stream();
    let stream = byte_stream
        .scan(String::new(), |buffer, chunk| {
            let chunk = match chunk {
                Ok(bytes) => bytes,
                Err(_) => return futures::future::ready(None),
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            let mut events = Vec::new();
            let mut event: Option<String> = None;
            let mut data = String::new();

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim_end_matches('\r').to_string();
                *buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    if !data.is_empty() {
                        events.push(SseEvent {
                            event: event.take(),
                            data: std::mem::take(&mut data),
                        });
                    } else {
                        event = None;
                        data.clear();
                    }
                    continue;
                }
                if let Some(value) = line.strip_prefix("event:") {
                    event = Some(value.trim().to_string());
                } else if let Some(value) = line.strip_prefix("data:") {
                    if !data.is_empty() {
                        data.push('\n');
                    }
                    data.push_str(value.trim_start());
                }
                // ":" prefixed lines are SSE comments; ignored.
            }
            futures::future::ready(Some((events, ())))
        })
        .flat_map(|(events, ())| futures::stream::iter(events));

    Ok(Box::pin(stream))
}

/// Build a [`ProviderError::Http`] from a failed response.
pub async fn http_error(response: Response) -> ProviderError {
    let status = response.status().as_u16();
    let body = response.text().await.unwrap_or_default();
    if status == 401 || status == 403 {
        ProviderError::Auth(body)
    } else {
        ProviderError::Http { status, body }
    }
}
