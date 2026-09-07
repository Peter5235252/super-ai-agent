//! SpaceXAI (Grok) model metadata as of September 2026.
//!
//! xAI rebranded as SpaceXAI on July 6, 2026 (SpaceX merger); Grok kept its
//! name, API (`api.x.ai`) and endpoints. Grok is a legitimate frontier
//! pick for agents: 4.6 leads GDPVal-AA v2 and FrontierCode and ties the
//! AA Intelligence Index at 61 with GPT-5.6 Sol. Pricing verified Sept 2026.

use provider_api::{CapabilitySet, ModelInfo, ProviderKind};

fn caps() -> CapabilitySet {
    let mut c = CapabilitySet::chat_default();
    c.vision = true; // text + image input
    c.web_search = true; // native web search + X search tools
    c
}

pub fn known_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "grok-4.6".into(),
            provider: ProviderKind::Xai,
            display_name: "Grok 4.6".into(),
            capabilities: caps(),
            context_window: Some(500_000),
            max_output_tokens: None, // no text output limit
            input_price_per_mtok: Some(2.0),
            output_price_per_mtok: Some(6.0),
            knowledge_cutoff: Some("Feb 2026".into()),
            notes: Some(
                "Flagship (Aug 12, 2026). Reasoning low/medium/high/xhigh; native web + X \
                 search; top agentic scores (GDPVal-AA v2, FrontierCode). Cached input \
                 $0.50/MTok; 2x long-context rates past 200K tokens."
                    .into(),
            ),
        },
        ModelInfo {
            id: "grok-4.5".into(),
            provider: ProviderKind::Xai,
            display_name: "Grok 4.5".into(),
            capabilities: caps(),
            context_window: Some(500_000),
            max_output_tokens: None,
            input_price_per_mtok: Some(2.0),
            output_price_per_mtok: Some(6.0),
            knowledge_cutoff: Some("Feb 2026".into()),
            notes: Some(
                "Previous flagship (Jul 2026); same $2/$6 as 4.6 with cheaper cached \
                 input ($0.30/MTok). Fine for existing deployments."
                    .into(),
            ),
        },
        ModelInfo {
            id: "grok-4.3".into(),
            provider: ProviderKind::Xai,
            display_name: "Grok 4.3".into(),
            capabilities: caps(),
            context_window: Some(1_000_000),
            max_output_tokens: None,
            input_price_per_mtok: Some(1.25),
            output_price_per_mtok: Some(2.5),
            knowledge_cutoff: None,
            notes: Some(
                "Cost-efficient general reasoning with 1M context; web/X search included.".into(),
            ),
        },
        ModelInfo {
            id: "grok-build-0.1".into(),
            provider: ProviderKind::Xai,
            display_name: "Grok Build 0.1".into(),
            capabilities: caps(),
            context_window: Some(256_000),
            max_output_tokens: None,
            input_price_per_mtok: Some(1.0),
            output_price_per_mtok: Some(2.0),
            knowledge_cutoff: None,
            notes: Some("Coding specialist; cheaper than 4.3 at higher SWE quality.".into()),
        },
    ]
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
