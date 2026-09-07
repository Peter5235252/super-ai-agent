//! xAI (Grok) model metadata as of September 2026.

use provider_api::{CapabilitySet, ModelInfo, ProviderKind};

pub fn known_models() -> Vec<ModelInfo> {
    let mut caps = CapabilitySet::chat_default();
    caps.vision = true; // text + image input
    caps.web_search = true; // native web search + X search tools
    vec![ModelInfo {
        id: "grok-4.6".into(),
        provider: ProviderKind::Xai,
        display_name: "Grok 4.6".into(),
        capabilities: caps,
        context_window: Some(500_000),
        max_output_tokens: None, // no text output limit
        input_price_per_mtok: Some(2.0),
        output_price_per_mtok: Some(6.0),
        knowledge_cutoff: Some("Jan 2026".into()),
        notes: Some(
            "Flagship (Aug 2026). Reasoning effort low/medium/high/xhigh; native web search \
                 and X search; recommends prompt_cache_key for long agent loops."
                .into(),
        ),
    }]
}

pub fn resolve_model(id: &str) -> ModelInfo {
    known_models()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| ModelInfo {
            id: id.to_string(),
            provider: ProviderKind::Xai,
            display_name: id.to_string(),
            capabilities: CapabilitySet::chat_default(),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: None,
            output_price_per_mtok: None,
            knowledge_cutoff: None,
            notes: Some("Unlisted Grok model; using generic capability defaults.".into()),
        })
}
