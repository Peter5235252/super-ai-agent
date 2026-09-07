//! SpaceXAI (Grok) provider adapter.
//!
//! Grok exposes the OpenAI-compatible Responses API, so this crate reuses
//! `provider-openai`'s streaming client with SpaceXAI's endpoint and model
//! table. (xAI rebranded as SpaceXAI in July 2026; the API is unchanged.)
#![forbid(unsafe_code)]

pub mod models;

use async_trait::async_trait;
use provider_api::{
    AgentRequest, ModelInfo, ModelProvider, ProviderEventStream, ProviderKind, Result,
};
use provider_openai::ResponsesClient;
use secrecy::SecretString;

pub const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";
pub const DEFAULT_MODEL: &str = "grok-4.6";

pub struct XaiProvider(pub ResponsesClient);

impl XaiProvider {
    pub fn new(
        api_key: SecretString,
        base_url: Option<String>,
        default_model: Option<String>,
    ) -> Self {
        Self(ResponsesClient::new(
            api_key,
            base_url,
            default_model,
            ProviderKind::Xai,
            models::known_models,
        ))
    }
}

#[async_trait]
impl ModelProvider for XaiProvider {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grok_metadata_current() {
        let m = models::resolve_model("grok-4.6");
        assert_eq!(m.context_window, Some(500_000));
        assert!(m.capabilities.web_search);
        assert_eq!(m.input_price_per_mtok, Some(2.0));
        assert_eq!(m.output_price_per_mtok, Some(6.0));
        let ids: Vec<String> = models::known_models().into_iter().map(|m| m.id).collect();
        assert!(ids.contains(&"grok-4.5".to_string()));
        assert!(ids.contains(&"grok-4.3".to_string()));
        assert!(ids.contains(&"grok-build-0.1".to_string()));
    }
}
