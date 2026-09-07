import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { AgentEvent, DiagRow, MessageRow, ModelInfo, ProviderRow, SessionRow } from "./types";

export const api = {
  createSession: (title: string | null, workspace: string | null) =>
    invoke<SessionRow>("create_session", { title, workspace }),
  listSessions: () => invoke<SessionRow[]>("list_sessions"),
  deleteSession: (id: string) => invoke<void>("delete_session", { id }),
  getMessages: (sessionId: string) => invoke<MessageRow[]>("get_messages", { sessionId }),
  bindSessionProvider: (sessionId: string, providerName: string, model: string) =>
    invoke<void>("bind_session_provider", { sessionId, providerName, model }),
  setSessionEffort: (sessionId: string, effort: string | null) =>
    invoke<void>("set_session_effort", { sessionId, effort }),
  setMode: (mode: string) => invoke<string>("set_mode", { mode }),
  setAutonomy: (risk: string) => invoke<void>("set_autonomy", { risk }),
  listProviders: () => invoke<ProviderRow[]>("list_providers"),
  saveProvider: (
    name: string,
    kind: string,
    baseUrl: string | null,
    defaultModel: string | null,
    apiKey: string | null,
  ) => invoke<string>("save_provider", { name, kind, baseUrl, defaultModel, apiKey }),
  deleteProvider: (name: string) => invoke<string>("delete_provider", { name }),
  hasProviderKey: (name: string) => invoke<boolean>("has_provider_key", { name }),
  testProvider: (name: string) => invoke<string>("test_provider", { name }),
  knownModels: (kind: string) => invoke<ModelInfo[]>("known_models", { kind }),
  liveModels: (name: string) => invoke<ModelInfo[]>("live_models", { name }),
  sendMessage: (sessionId: string, text: string) =>
    invoke<string>("send_message", { sessionId, text }),
  approve: (approvalId: string, allow: boolean) =>
    invoke<boolean>("approve", { approvalId, allow }),
  runDiagnostics: (sessionId: string | null) =>
    invoke<DiagRow[]>("run_diagnostics", { sessionId }),
  getSetting: (key: string) => invoke<string | null>("get_setting", { key }),
  setSetting: (key: string, value: string) => invoke<void>("set_setting", { key, value }),
};

export function onAgentEvent(cb: (e: AgentEvent) => void): Promise<UnlistenFn> {
  return listen<AgentEvent>("agent-event", (event) => cb(event.payload));
}
