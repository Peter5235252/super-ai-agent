//! Anthropic model metadata as of September 2026.
//!
//! Specs are from the Claude Platform docs (sonnet-5 overview page).
//! Community-reported Opus 5.1 / Sonnet 5.1 are not in the official docs
//! yet and are intentionally absent; unknown ids fall back to generic
//! metadata automatically.

use provider_api::{CapabilitySet, ModelInfo, ProviderKind};

fn claude_caps(vision: bool) -> CapabilitySet {
    let mut c = CapabilitySet::chat_default();
    c.vision = vision;
    c
}

pub fn known_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "claude-fable-5-1".into(),
            provider: ProviderKind::Anthropic,
            display_name: "Claude Fable 5.1".into(),
            capabilities: claude_caps(true),
            context_window: Some(1_000_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(10.0),
            output_price_per_mtok: Some(50.0),
            knowledge_cutoff: Some("Jun 2026".into()),
            notes: Some(
                "Most advanced Anthropic model (Sep 2026). Adaptive thinking always on; built \
                 for long-running agentic coding and multistep research. Cache reads at a \
                 quarter of normal cost."
                    .into(),
            ),
        },
        ModelInfo {
            id: "claude-opus-5".into(),
            provider: ProviderKind::Anthropic,
            display_name: "Claude Opus 5".into(),
            capabilities: claude_caps(true),
            context_window: Some(1_000_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(5.0),
            output_price_per_mtok: Some(25.0),
            knowledge_cutoff: Some("May 2026".into()),
            notes: Some("Recommended default for most API workloads (Jul 2026).".into()),
        },
        ModelInfo {
            id: "claude-sonnet-5".into(),
            provider: ProviderKind::Anthropic,
            display_name: "Claude Sonnet 5".into(),
            capabilities: claude_caps(true),
            context_window: Some(1_000_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(2.0),
            output_price_per_mtok: Some(10.0),
            knowledge_cutoff: Some("Jan 2026".into()),
            notes: Some(
                "Fast Sonnet-tier model (Jun 2026). Drop-in upgrade over Sonnet 4.6; adaptive \
                 thinking on by default; temperature/top_p/top_k must stay at defaults."
                    .into(),
            ),
        },
        ModelInfo {
            id: "claude-haiku-4-5".into(),
            provider: ProviderKind::Anthropic,
            display_name: "Claude Haiku 4.5".into(),
            capabilities: claude_caps(true),
            context_window: Some(200_000),
            max_output_tokens: Some(64_000),
            input_price_per_mtok: Some(1.0),
            output_price_per_mtok: Some(5.0),
            knowledge_cutoff: Some("Feb 2025".into()),
            notes: Some("Cheapest/fastest Claude tier.".into()),
        },
    ]
}

pub fn resolve_model(id: &str) -> ModelInfo {
    known_models()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| ModelInfo {
            id: id.to_string(),
            provider: ProviderKind::Anthropic,
            display_name: id.to_string(),
            capabilities: CapabilitySet::chat_default(),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: None,
            output_price_per_mtok: None,
            knowledge_cutoff: None,
            notes: Some("Unlisted Claude model; using generic capability defaults.".into()),
        })
}
