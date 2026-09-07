# Super-AI

A Windows 11 desktop AI agent built in memory-safe Rust — 100% native
(egui, no WebView, no dev server). Bring Your Own API Key (BYOK) — OpenAI,
Anthropic, xAI (Grok), Mistral, Google Gemini — plus local LLMs via Ollama
or any OpenAI-compatible server (LM Studio, vLLM) — with live-streaming
chat, an event-driven agent runtime, workspace-confined filesystem tools, a
terminal tool (PowerShell/cmd), a policy engine with human approval gates,
and SQLite persistence.

> **Status:** Phase 0/1 of the master roadmap — a working vertical slice:
> provider abstraction → streaming event bus → agent loop → typed tools →
> policy/approval → persistence → desktop shell.

## What works today

- **BYOK providers**: OpenAI (GPT-6 Astra, GPT-5.6 Sol/Terra/Luna/Cyber),
  Anthropic (Claude Fable 5.1, Opus 5, Sonnet 5, Haiku 4.5), xAI (Grok 4.6),
  Mistral (Large, Medium, Small, Codestral, Devstral), Google Gemini
  (3 Pro/Flash, 2.5 Pro/Flash via the OpenAI-compat shim), Ollama
  (llama4:scout, qwen3, llama3.1, mistral, …) and generic local servers
  (LM Studio :1234, vLLM :8000 — no key needed).
  Keys are stored in the **Windows Credential Manager** via `keyring` —
  never in SQLite, logs, or the model prompt.
- **Live streaming** chat: token deltas stream into the UI as they arrive.
- **Agent loop**: the model can call `fs.list`, `fs.read`, `fs.write` and
  `process.run` (PowerShell/cmd with timeout, stdout/stderr capture) inside
  a workspace; results feed back for another turn (up to a turn budget, with
  retry/backoff on stream failures).
- **Policy engine**: low-risk operations run automatically; medium risk
  (`fs.write`) and HIGH risk (`process.run`) raise an approval card with
  exact arguments, risk class and reason — Allow once / Deny. Credential
  tools are hard-denied by policy.
- **Event-driven runtime**: every step (`task_created`, `model_delta`,
  `tool_requested`, `approval_requested`, `tool_output`, `task_completed`, …)
  is broadcast to the UI **and** persisted as flight-recorder rows.
- **SQLite persistence** (versioned migrations): sessions, messages, events,
  provider config, settings.
- **Workspace confinement**: filesystem tools canonicalize every path and
  reject `..`/symlink escapes outside the workspace root (unit-tested).

## Model catalog (September 2026)

| Provider | Model | Context | $ in/out per MTok | Notes |
|---|---|---|---|---|
| OpenAI | `gpt-6-astra` | — | — | Newest frontier (Sep 2026); SOTA computer use / browsing / SWE |
| OpenAI | `gpt-5.6-sol` | — | 5 / 30 | Flagship 5.6; best coding/agentic |
| OpenAI | `gpt-5.6-terra` | — | 2.50 / 15 | Mid tier |
| OpenAI | `gpt-5.6-luna` | — | 1 / 6 | Fast/cheap |
| OpenAI | `gpt-5.6-cyber` | — | — | Daybreak cyber variant |
| Anthropic | `claude-fable-5-1` | 1M | 10 / 50 | Most advanced; adaptive thinking |
| Anthropic | `claude-opus-5` | 1M | 5 / 25 | Recommended default |
| Anthropic | `claude-sonnet-5` | 1M | 2 / 10 | Fast; temperature must stay default |
| Anthropic | `claude-haiku-4-5` | 200K | 1 / 5 | Cheapest tier |
| xAI | `grok-4.6` | 500K | 2 / 6 | Native web/X search; reasoning low→xhigh |
| Mistral | `mistral-large-latest` | 128K | 3 / 9 | Flagship; EU data residency |
| Mistral | `mistral-medium-latest` | — | 2.70 / 8.10 | Reliable function calling |
| Mistral | `mistral-small-latest` | — | 0.20 / 0.60 | Price-performance pick |
| Mistral | `codestral-latest` | — | 0.30 / 0.90 | Code-specialized; best for coding agents |
| Mistral | `devstral-latest` | — | — | Agentic coding; confirm id in Mistral docs |
| Google | `gemini-3-pro` | — | — | Frontier reasoning; confirm id in AI Studio |
| Google | `gemini-3-flash` | — | — | Fast/cheap reasoning tier |
| Google | `gemini-2.5-pro` | — | — | Long-context workhorse |
| Google | `gemini-2.5-flash` | — | — | High-volume extraction tier |
| Ollama | `llama4:scout` | — | 0 (local) | Best local agentic quality (~12 GB) |
| Ollama | `qwen3:14b` / `qwen3:8b` | — | 0 (local) | Strong local tool calling |
| Ollama | `llama3.1:8b` | — | 0 (local) | Legacy, solid tools |
| Local | any `/v1/models` id | — | 0 (local) | LM Studio, vLLM, llama.cpp — exact id from server |

Unlisted/unknown model ids resolve to generic metadata automatically, so new
releases work without a code change. Opus 5.1 / Sonnet 5.1 are community-
reported but not yet in the official docs, so they are intentionally absent.

## Architecture

```text
app-ui (native egui, single binary: super-ai-native.exe)
  sessions/chat · streaming deltas · activity feed · approvals · providers
                         │
      agent-runtime (Agent loop + ToolRegistry + AgentEvent bus)
      policy-engine (risk scoring + approval gates)
      tool-core / tool-filesystem (workspace-confined) / tool-process (PowerShell/cmd)
      provider-api ← provider-openai / -anthropic / -xai / -compat (mistral/gemini/ollama/local)
      persistence (sqlx SQLite)   secrets (Credential Manager)
```

All crates use `#![forbid(unsafe_code)]`. The agent brain never depends on a
vendor SDK; providers convert the canonical `Message` format to their wire
format and back.

## Run it

Prerequisites: stable Rust, Windows 11. No Node, no WebView, no dev server.

```powershell
cargo run -p app-ui                                  # dev
cargo build --release -p app-ui                      # release exe at target\release\super-ai-native.exe
Copy-Item target\release\super-ai-native.exe Super-AI.exe -Force   # double-clickable copy
```

Local LLMs: start Ollama (`ollama serve` + `ollama pull qwen3:8b`) or LM
Studio's server, then add an `ollama` / `local` provider (no key needed)
and **Test connection**.

1. Click **Providers & keys** → add e.g. `mistral` with your key (stored in
   Credential Manager; skip the key for Ollama/local) → **Test connection**.
2. **+ New session** with an optional workspace path.
3. Bind a provider + model in the header, then chat. Ask it to
   "list the files in the workspace", "run the tests", or "create a file" —
   writes and terminal commands raise an approval card.

## Checks

```bash
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI (`.github/workflows/ci.yml`) runs all three.
`cargo audit` / `cargo deny` should be added once installed.

## Roadmap status

Native shell done (chat streaming, providers/approvals, workspace +
activity = old M1–M3). Phase 0/1 (providers, streaming, event bus, agent
loop, tools, persistence) plus Phase 3 basics (`process.run`: PowerShell/
cmd with timeouts + HIGH-risk approval) are in place. Next: **Phase 2**
filesystem intelligence (search, project detection, diffs), **Phase 4**
workspace ACLs + audit UX, **Phase 5** OS-level sandboxing (AppContainer /
Job Objects / restricted tokens). See `docs/ARCHITECTURE.md` and the master
roadmap in `docs/ROADMAP.md` for the full plan.