# Super-AI master roadmap

Status legend: ✅ done · 🔨 in progress · ⬜ planned

## Phases

| # | Phase | Status |
|---|-------|--------|
| 0 | Architecture prototype — Tauri shell, Rust backend, provider abstraction, streaming events, basic chat, SQLite, settings | ✅ (shell being replaced by native egui UI — see NATIVE-UI-PLAN.md) |
| 1 | Real agent runtime — tool schemas, tool calling, agent loop, event system, task persistence, retries | ✅ |
| 2 | Workspace intelligence — filesystem, Git, project detection, code search, diff engine, workspace memory | ⬜ |
| 3 | Terminal agent — PowerShell/cmd/Git Bash/WSL, streaming processes, timeouts, interactive commands | ⬜ |
| 4 | Approval + policy engine — permission classes, approval UX, workspace ACLs, risk scoring, audit log | 🔨 (core engine ✅, full UX pending) |
| 5 | Sandboxing — restricted processes, Job Objects, AppContainer, filesystem/network restrictions, mitigations | ⬜ |
| 6 | Browser + web — web search, URL fetch, browser automation, screenshots, page extraction | ⬜ |
| 7 | Computer control — desktop screenshot, window management, mouse/keyboard, accessibility, GUI automation | ⬜ |
| 8 | MCP — discovery, stdio + Streamable HTTP, tool registry, server permissions/isolation (2026 spec) | ⬜ |
| 9 | Multi-agent orchestration — sub-agents, delegation, parallel tasks, budgets | ⬜ |
| 10 | Advanced memory/context — semantic memory, project/global memory, ranking, compaction, provenance | ⬜ |
| 11 | Verification + recovery — automatic testing, verification agents, rollback, checkpoints, resume | ⬜ |
| 12 | Advanced autonomy — autonomy profiles, goal-driven tasks, task DAGs, scheduling, background agents | ⬜ |

## Non-goals (reminders)

- No single enormous Rust module; UI never talks to providers directly.
- No raw API keys in prompts, logs, DB or config; no prompt-only security.
- No "sandboxed" claims without OS-enforced boundaries.
- No vendor lock-in; every capability is a typed, policy-checked tool.

## Definition of done (beta)

A fresh Windows 11 install can: install → add key → pick model → open project
→ understand → plan → read/edit files → run tests → observe + fix failures →
verify → browse web → use MCP tools → approve risky actions → run inside
sandbox limits → keep task history → resume interrupted tasks → delegate →
produce a final report.