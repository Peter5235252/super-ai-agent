//! Model catalogs for OpenAI-compatible providers (September 2026).
//!
//! Mistral (La Plateforme), Google Gemini (OpenAI-compat shim), Ollama and
//! generic local servers (LM Studio, vLLM, llama.cpp) all speak
//! `/chat/completions` with SSE streaming and `tools`/`tool_calls`.
//! Entries without confirmed specs carry `None` rather than invented
//! numbers; verify live IDs against each vendor's docs — unlisted IDs
//! resolve to generic metadata automatically.

use provider_api::{CapabilitySet, ModelInfo, ProviderKind};

fn caps(tool: bool) -> CapabilitySet {
    let mut c = CapabilitySet::chat_default();
    c.tool_calling = tool;
    c
}

// ---------------------------------------------------------------------------
// Mistral
// ---------------------------------------------------------------------------

/// Known Mistral API models; `list_models` merges these with `/models`.
pub fn known_models_mistral() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "mistral-large-latest".into(),
            provider: ProviderKind::Mistral,
            display_name: "Mistral Large".into(),
            capabilities: caps(true),
            context_window: Some(128_000),
            max_output_tokens: None,
            input_price_per_mtok: Some(3.0),
            output_price_per_mtok: Some(9.0),
            knowledge_cutoff: None,
            notes: Some(
                "Flagship API model (Large 2 family); frontier quality, EU data residency. \
                 Mid-2026 list pricing; confirm in Mistral docs."
                    .into(),
            ),
        },
        ModelInfo {
            id: "mistral-medium-latest".into(),
            provider: ProviderKind::Mistral,
            display_name: "Mistral Medium".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(2.7),
            output_price_per_mtok: Some(8.1),
            knowledge_cutoff: None,
            notes: Some("Mid tier; reliable function calling for agentic loops.".into()),
        },
        ModelInfo {
            id: "mistral-small-latest".into(),
            provider: ProviderKind::Mistral,
            display_name: "Mistral Small".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(0.2),
            output_price_per_mtok: Some(0.6),
            knowledge_cutoff: None,
            notes: Some("Price-performance champion for high-volume tasks.".into()),
        },
        ModelInfo {
            id: "codestral-latest".into(),
            provider: ProviderKind::Mistral,
            display_name: "Codestral".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(0.3),
            output_price_per_mtok: Some(0.9),
            knowledge_cutoff: None,
            notes: Some("Code-specialized model; best Mistral pick for coding agents.".into()),
        },
        ModelInfo {
            id: "devstral-latest".into(),
            provider: ProviderKind::Mistral,
            display_name: "Devstral".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: None,
            output_price_per_mtok: None,
            knowledge_cutoff: None,
            notes: Some("Agentic coding model; confirm id/pricing in Mistral docs.".into()),
        },
    ]
}

/// Resolve a Mistral model id, falling back to generic metadata.
pub fn resolve_mistral(id: &str) -> ModelInfo {
    known_models_mistral()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| generic(id, ProviderKind::Mistral))
}

// ---------------------------------------------------------------------------
// Gemini (via Google's OpenAI-compat shim)
// ---------------------------------------------------------------------------

/// Known Gemini models (September 2026); confirm exact versioned IDs in
/// Google AI Studio — Google revises them on its own cadence.
pub fn known_models_gemini() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "gemini-3.8-flash".into(),
            provider: ProviderKind::Google,
            display_name: "Gemini 3.8 Flash".into(),
            capabilities: caps(true),
            context_window: Some(1_048_576),
            max_output_tokens: Some(65_536),
            input_price_per_mtok: Some(0.75),
            output_price_per_mtok: Some(3.75),
            knowledge_cutoff: Some("Mar 2026".into()),
            notes: Some(
                "Newest Flash (Sep 2, 2026); best for coding/agents. Intro rate through \
                 Dec 31, 2026."
                    .into(),
            ),
        },
        ModelInfo {
            id: "gemini-3.7-flash".into(),
            provider: ProviderKind::Google,
            display_name: "Gemini 3.7 Flash".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(0.75),
            output_price_per_mtok: Some(3.75),
            knowledge_cutoff: None,
            notes: Some(
                "Previous Flash (Aug 2026); same intro rate through Dec 31, 2026. Confirm \
                 specs in AI Studio."
                    .into(),
            ),
        },
        ModelInfo {
            id: "gemini-3.6-flash".into(),
            provider: ProviderKind::Google,
            display_name: "Gemini 3.6 Flash".into(),
            capabilities: caps(true),
            context_window: None,
            max_output_tokens: None,
            input_price_per_mtok: Some(0.75),
            output_price_per_mtok: Some(3.75),
            knowledge_cutoff: None,
            notes: Some(
                "July 2026 Flash; same intro rate through Dec 31, 2026. Confirm specs in \
                 AI Studio."
                    .into(),
            ),
        },
        ModelInfo {
            id: "gemini-3.1-pro-preview".into(),
            provider: ProviderKind::Google,
            display_name: "Gemini 3.1 Pro".into(),
            capabilities: caps(true),
            context_window: Some(1_048_576),
            max_output_tokens: Some(65_536),
            input_price_per_mtok: Some(2.5),
            output_price_per_mtok: Some(15.0),
            knowledge_cutoff: Some("Jan 2025".into()),
            notes: Some(
                "Flagship reasoning (Feb 2026); paid tier only. Confirm id/pricing in AI \
                 Studio."
                    .into(),
            ),
        },
    ]
}

/// Resolve a Gemini model id, falling back to generic metadata.
pub fn resolve_gemini(id: &str) -> ModelInfo {
    known_models_gemini()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| generic(id, ProviderKind::Google))
}

// ---------------------------------------------------------------------------
// Ollama (local)
// ---------------------------------------------------------------------------

/// Popular local tags that support tool calling (2026). Availability depends
/// entirely on what you pulled (`ollama pull <tag>`); anything else you
/// pulled resolves generically via the live `/v1/models` listing.
pub fn known_models_ollama() -> Vec<ModelInfo> {
    vec![
        local_model(
            "llama4:scout",
            "Llama 4 Scout",
            "Best agentic quality ~12 GB; reliable tools.",
        ),
        local_model(
            "qwen3:14b",
            "Qwen3 14B",
            "Strong local tool calling; needs ~10 GB.",
        ),
        local_model("qwen3:8b", "Qwen3 8B", "Good tools on modest hardware."),
        local_model(
            "llama3.1:8b",
            "Llama 3.1 8B",
            "Legacy but solid tool support.",
        ),
        local_model(
            "mistral:7b",
            "Mistral 7B",
            "Laptop-friendly; weak on multi-step tool loops.",
        ),
        local_model(
            "llama3.2:3b",
            "Llama 3.2 3B",
            "Lightweight; unreliable tool calling.",
        ),
    ]
}

/// Resolve an Ollama model id, falling back to generic metadata.
pub fn resolve_ollama(id: &str) -> ModelInfo {
    known_models_ollama()
        .into_iter()
        .find(|m| m.id == id)
        .unwrap_or_else(|| generic(id, ProviderKind::Ollama))
}

fn local_model(id: &str, display: &str, notes: &str) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        provider: ProviderKind::Ollama,
        display_name: display.to_string(),
        capabilities: caps(true),
        context_window: None,
        max_output_tokens: None,
        input_price_per_mtok: Some(0.0),
        output_price_per_mtok: Some(0.0),
        knowledge_cutoff: None,
        notes: Some(format!("Runs on your machine. {notes}")),
    }
}

// ---------------------------------------------------------------------------
// Generic local server (LM Studio, vLLM, llama.cpp, ...)
// ---------------------------------------------------------------------------

/// No offline catalog: the live `/v1/models` listing is authoritative.
pub fn known_models_local() -> Vec<ModelInfo> {
    Vec::new()
}

/// Resolve any model id against the generic local metadata.
pub fn resolve_local(id: &str) -> ModelInfo {
    generic(id, ProviderKind::Local)
}

fn generic(id: &str, provider: ProviderKind) -> ModelInfo {
    ModelInfo {
        id: id.to_string(),
        provider,
        display_name: id.to_string(),
        capabilities: CapabilitySet::chat_default(),
        context_window: None,
        max_output_tokens: None,
        input_price_per_mtok: None,
        output_price_per_mtok: None,
        knowledge_cutoff: None,
        notes: Some("Unlisted model; using generic capability defaults.".into()),
    }
}
