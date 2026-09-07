# Super-AI architecture

## Layered design

```text
┌──────────────────────────────────────────────────────┐
│ Desktop UI (React + Vite in WebView2)                │
└───────────────────────┬──────────────────────────────┘
                        │  invoke() commands / "agent-event" channel
┌───────────────────────▼──────────────────────────────┐
│ src-tauri (Tauri 2 shell)                            │
│  commands.rs — the only IPC surface                  │
│  lib.rs — global event forwarder (UI + flight log)   │
└─────────────┬───────────────────────────┬────────────┘
              │                           │
┌─────────────▼───────────┐   ┌───────────▼───────────┐
│ agent-runtime           │   │ policy-engine          │
│ Agent loop (state       │   │ risk scoring           │
│ machine over broadcast  │   │ approval gates         │
│ channel)                │   │ hard denials           │
│ ToolRegistry            │   └───────────┬───────────┘
└──────┬──────────────────┘               │
       │                                  │
┌──────▼──────────────────────────────────▼──────────┐
│ tool-core / tool-filesystem                          │
│ typed tools · JSON schemas · workspace confinement  │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ provider-api ← provider-openai / -anthropic / -xai   │
│ ModelProvider trait · canonical Message · SSE parser │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│ persistence (sqlx/SQLite, versioned migrations)      │
│ secrets (Windows Credential Manager via keyring)     │
└──────────────────────────────────────────────────────┘
```

## Data flow for one task

1. `send_message` command loads session → provider config → API key from the
   OS credential store → builds `Arc<dyn ModelProvider>`.
2. Conversation history is read from SQLite and converted to the canonical
   `provider_api::Message` form.
3. A `TaskRequest` (with workspace-rooted filesystem tools) is spawned on the
   agent. The agent loop:
   - streams the model response, re-emitting `ModelDelta` chunks;
   - collects finished tool calls;
   - policy-checks each call (auto-allow ≤ LOW, approve MEDIUM+, hard-deny
     credential tools);
   - executes approved calls with timeouts, feeds results back as `Tool`
     messages, and repeats until no tool calls remain or the turn budget is
     exhausted.
4. Every `AgentEvent` is broadcast; the single forwarder in `src-tauri`
   relays it to the webview (`agent-event`) and appends a row to `events`
   (the flight recorder). Completed assistant messages are persisted to
   `messages`.

## Security posture (current)

- `#![forbid(unsafe_code)]` in every crate.
- API keys: `secrecy::SecretString` in memory; Credential Manager on disk;
  never serialized across IPC, into SQLite, or into logs.
- Filesystem tools canonicalize paths and reject traversal/symlink escapes
  outside the workspace root.
- The model receives only tool name/description/schema — never definitions
  carrying risk metadata.
- Prompt injection posture: file contents are data; the system prompt
  instructs the model accordingly (hard enforcement comes with the
  sanitizer/context layer in a later phase).

## Phase checklist

See `ROADMAP.md` for the 12-phase plan. Current: Phases 0–1 complete
(providers, streaming, event bus, agent loop, basic tools, persistence).
Next: Phase 2 (filesystem intelligence), 3 (process broker), 4 (approval UX +
workspace ACLs), 5 (OS sandbox: AppContainer / Job Objects / restricted
tokens).