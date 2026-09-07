# Native UI Migration Plan

**Goal:** replace the Tauri 2 + React/WebView2 shell with a fully native Rust
UI (no web components at all) — a single native binary, instant startup, no
webview process, and a product look that is deliberately different from
Codex/Claude Code shells.

**Status:** M0 scaffolded (`crates/app-ui`, egui/eframe 0.36, verified
compiling). The existing Tauri build stays as the fallback until parity.

---

## 1. Why native

| | Webview shell (current) | Native Rust UI (target) |
|---|---|---|
| Startup | WebView2 child process + HTML/CSS/JS boot | Direct window, first paint in ms |
| Memory | ~100–200 MB working set | Typically 30–60 MB |
| Binary | 17 MB exe + WebView2 dependency | ~10–15 MB, zero web runtime |
| Attack surface | Webview + IPC bridge + injected scripts | Immediate-mode UI, no DOM |
| Differentiation | "Another Electron-style app" | Genuinely distinct from Codex/Claude Code |

All 11 core crates (agent runtime, providers, policy, tools, persistence,
secrets) are **untouched** — they are already UI-agnostic. Only the shell
layer changes.

## 2. Framework decision

**Chosen: egui / eframe 0.36** (immediate mode, winit windowing, glow/OpenGL
renderer on Windows).

- Best fit for tool-heavy, constantly-updating UIs (streaming text, live
  activity feed, approval cards, later: terminal + capability graph).
- No retained widget tree to fight when the agent's state is a fast-moving
  event stream — the UI just re-renders from state each frame.
- Excellent panel/docking ecosystem: `egui_dock` for the multi-panel layout
  the roadmap calls for (Chat / Files / Terminal / Plan / Activity / Graph).
- Markdown + code rendering for chat: `egui_markdown` (or `egui_commonmark`),
  `egui_code_editor` for editable code views.
- Single-binary, `#![forbid(unsafe_code)]` friendly.

Rejected alternatives:

| Framework | Why not |
|---|---|
| iced | Nice retained-mode model, but slower iteration for this density of live widgets; weaker docking/terminal ecosystem today |
| Slint | DSL + GPL/commercial dual license is a business constraint for a BYOK product; less suited to streaming text density |
| Win32/WPF via windows-rs | The roadmap's "no giant module" rule and rich-output rendering argue against hand-rolling every widget |

## 3. Target architecture

```text
super-ai.exe (single native binary)
├─ app-ui (egui/eframe)          ← replaces src-tauri + frontend
│   ├─ chat / sessions panel
│   ├─ activity feed (flight recorder view)
│   ├─ approval center
│   ├─ providers/key management
│   └─ later: files, terminal, diff, plan, capability graph (egui_dock)
├─ agent-runtime (Agent loop, broadcast AgentEvent bus)   [unchanged]
├─ policy-engine · tool-core · tool-filesystem            [unchanged]
├─ provider-* · persistence · secrets · app-core          [unchanged]
```

Key mechanics:

- **No IPC.** The UI calls `Agent::spawn(...)`, `PolicyEngine::respond(...)`,
  `Db::...` directly. The global event forwarder (currently in `src-tauri`)
  becomes a small UI-side drain: subscribe to the broadcast channel, and in
  `update()` do `try_recv()` per frame, `ctx.request_repaint()` on new events.
- **Async bridge.** One tokio runtime lives in the binary. UI-initiated async
  work (send message, test connection, list models) is spawned and its
  results arrive back through the same event channel or a per-call oneshot —
  the UI never blocks.
- **Same event model.** `AgentEvent` stays the single source of truth; the
  flight recorder (SQLite `events` table) and message persistence keep
  working via the same drain logic that `forward_event` uses today.
- **Fonts.** Bundle a font covering Latin Extended (Hungarian ő/ű) +
  monospace for code/terminal; egui's defaults cover these but we pin them
  explicitly for consistency.

## 4. Milestones

| # | Milestone | Scope | Exit criteria |
|---|---|---|---|
| M0 | Skeleton | `crates/app-ui`: eframe window, dark theme, placeholder panels, workspace wiring | Compiles, opens a window (✅ done) |
| M1 | Chat streaming parity | Sessions list, messages, streaming deltas, send, markdown rendering | Type a message with a bound provider → tokens stream in |
| M2 | Providers + approvals | Provider modal (Credential Manager via `secrets`), test connection, model catalog, approval cards with Allow/Deny | Same UX as web build for BYOK + approvals |
| M3 | Workspace + activity | Workspace binding, tool activity feed, flight-recorder view, task summary chips | fs.read/fs.write round-trip visible in UI |
| M4 | Native extras | egui_dock panels: files, terminal (Phase 3 broker), plan, capability graph | Differentiator panels exist |
| M5 | Retire web shell | Remove `src-tauri` + `frontend` from workspace; installer via NSIS/MSI (Tauri bundler can be dropped; use `cargo-wix`/NSIS directly) | `cargo build --release` produces the only binary |

## 5. Risks

- **Markdown fidelity** — egui_markdown covers the roadmap's needs (code,
  tables, lists); exotic HTML in model output is rendered as plain text
  (acceptable, arguably safer).
- **Input methods** — egui text editing handles IME on Windows; verify
  Hungarian input early (M1).
- **Rendering throughput** — batched text appends (already how the runtime
  emits deltas) keep per-frame cost low; `request_repaint_after` bounds CPU.
- **Time** — re-implementing the shell UI is real work; mitigated by keeping
  the web build as fallback until M3 parity.

## 6. Immediate next step

M1: wire `app-ui` to `agent-runtime` — sessions via `Db`, chat streaming via
the broadcast drain, send via `Agent::spawn`. The crate skeleton is ready in
`crates/app-ui` with the layout structure in place.