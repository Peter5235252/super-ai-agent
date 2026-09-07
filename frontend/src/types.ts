export interface SessionRow {
  id: string;
  title: string;
  workspace: string | null;
  provider_name: string | null;
  model: string | null;
  status: string;
  reasoning_effort: string | null;
  created_at: number;
  updated_at: number;
}

export interface MessageRow {
  id: string;
  session_id: string;
  role: string;
  content: string;
  tool_calls: string | null;
  tool_call_id: string | null;
  reasoning: string | null;
  created_at: number;
}

export interface ProviderRow {
  name: string;
  kind: string;
  base_url: string | null;
  default_model: string | null;
  is_default: boolean;
  created_at: number;
  updated_at: number;
}

export interface ModelInfo {
  id: string;
  provider: string;
  display_name: string;
  capabilities: Record<string, boolean>;
  context_window: number | null;
  max_output_tokens: number | null;
  input_price_per_mtok: number | null;
  output_price_per_mtok: number | null;
  knowledge_cutoff: string | null;
  notes: string | null;
}

export interface TaskSummary {
  task_id: string;
  session_id: string;
  turns: number;
  tool_calls: number;
  input_tokens: number;
  output_tokens: number;
  final_text: string;
  status: string;
}

export interface DiagRow {
  name: string;
  ok: boolean | null;
  detail: string;
}

export interface ApprovalCard {
  approval_id: string;
  task_id: string;
  tool: string;
  args: unknown;
  risk: string;
  reason: string;
}

export type AgentEvent =
  | { type: "task_created"; task_id: string; session_id: string; provider: string; model: string }
  | { type: "reasoning_started"; task_id: string; session_id: string }
  | { type: "model_delta"; task_id: string; session_id: string; text: string }
  | { type: "reasoning_delta"; task_id: string; session_id: string; text: string }
  | { type: "tool_requested"; task_id: string; session_id: string; tool: string; args: unknown }
  | { type: "tool_started"; task_id: string; session_id: string; tool: string }
  | {
      type: "tool_output";
      task_id: string;
      session_id: string;
      tool: string;
      tool_call_id: string;
      output: string;
      truncated: boolean;
    }
  | {
      type: "approval_requested";
      task_id: string;
      session_id: string;
      approval_id: string;
      tool: string;
      args: unknown;
      risk: string;
      reason: string;
    }
  | {
      type: "approval_resolved";
      task_id: string;
      session_id: string;
      approval_id: string;
      decision: string;
    }
  | {
      type: "message_completed";
      task_id: string;
      session_id: string;
      text: string;
      reasoning: string;
      tool_calls: unknown[];
    }
  | { type: "task_failed"; task_id: string; session_id: string; error: string }
  | { type: "task_completed"; task_id: string; session_id: string; summary: TaskSummary };
