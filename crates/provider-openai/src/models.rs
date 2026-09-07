//! OpenAI model metadata as of September 2026.
//!
//! Pricing and context figures are from OpenAI's announcements; entries
//! without confirmed specs carry `None` rather than invented numbers.

use provider_api::{CapabilitySet, ModelInfo, ProviderKind};

fn caps(
    text: bool,
    vision: bool,
    tool: bool,
    web: bool,
    computer: bool,
    reasoning: bool,
) -> CapabilitySet {
    let mut c = CapabilitySet::chat_default();
    c.vision = vision;
    c.tool_calling = tool;
    c.web_search = web;
    c.computer_use = computer;
    c.reasoning = reasoning;
    c.text_generation = text;
    c
}

/// Known models; `list_models` merges these with the live API's response.
pub fn known_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gpt-6-astra".into(),
            provider: ProviderKind::OpenAI,
            display_name: "GPT-6 Astra".into(),
            capabilities: caps(true, true, true, true, true, true),
            context_window: Some(1_050_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(10.0),
            output_price_per_mtok: Some(50.0),
            knowledge_cutoff: Some("Apr 2026".into()),
            notes: Some(
                "Flagship (Sep 3, 2026); SOTA computer use, browsing, SWE. Staged \
                 Trusted-Access rollout; always reasons (no 'none' effort)."
                    .into(),
            ),
        },
        ModelInfo {
            id: "gpt-5.6-sol".into(),
            provider: ProviderKind::OpenAI,
            display_name: "GPT-5.6 Sol".into(),
            capabilities: caps(true, true, true, true, false, true),
            context_window: Some(1_050_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(4.0),
            output_price_per_mtok: Some(20.0),
            knowledge_cutoff: Some("Feb 2026".into()),
            notes: Some(
                "Flagship 5.6 (Jul 2026); best coding/agentic of the family. $4/$20 promo \
                 through Nov 21, 2026 (list $5/$30)."
                    .into(),
            ),
        },
        ModelInfo {
            id: "gpt-5.6-terra".into(),
            provider: ProviderKind::OpenAI,
            display_name: "GPT-5.6 Terra".into(),
            capabilities: caps(true, true, true, true, false, true),
            context_window: Some(1_050_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(2.0),
            output_price_per_mtok: Some(12.0),
            knowledge_cutoff: Some("Feb 2026".into()),
            notes: Some("Mid-tier 5.6; balanced production tier since the Jul 30 cut.".into()),
        },
        ModelInfo {
            id: "gpt-5.6-luna".into(),
            provider: ProviderKind::OpenAI,
            display_name: "GPT-5.6 Luna".into(),
            capabilities: caps(true, true, true, true, false, true),
            context_window: Some(1_050_000),
            max_output_tokens: Some(128_000),
            input_price_per_mtok: Some(0.2),
            output_price_per_mtok: Some(1.2),
            knowledge_cutoff: Some("Feb 2026".into()),
            notes: Some(
                "Fastest/cheapest 5.6; permanent price since the Jul 30 cut (-80%).".into(),
            ),
        },
        ModelInfo {
            id: "gpt-5.6-cyber".into(),
            provider: ProviderKind::OpenAI,
            display_name: "GPT-5.6 Cyber".into(),
            capabilities: caps(true, true, true, true, false, true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(12.5),
            output_price_per_mtok: Some(75.0),
            knowledge_cutoff: Some("Jun 2026".into()),
            notes: Some("Controlled-access cyber variant (Daybreak program, Aug 2026).".into()),
        },
    ]
}

/// Resolve a model id (e.g. from the live API) against known metadata,
/// falling back to a generic entry so brand-new models still work.
pub fn resolve_model(id: &str) -> ModelInfo {
    known_models()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| ModelInfo {
            id: id.to_string(),
            provider: ProviderKind::OpenAI,
            display_name: id.to_string(),
            capabilities: CapabilitySet::chat_default(),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: None,
            output_price_per_mtok: None,
            knowledge_cutoff: None,
            notes: Some("Unlisted model; using generic capability defaults.".into()),
        })
}
