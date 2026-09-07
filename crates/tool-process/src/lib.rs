//! Terminal tool: run PowerShell / cmd commands for the agent.
//!
//! `process.run` executes one command with a bounded timeout and returns
//! the exit code plus truncated stdout/stderr. The working directory is
//! always confined inside the workspace root (`..` escapes and symlinks
//! leaving the root are rejected, same rule as the filesystem tools).
//!
//! Safety comes from the policy engine, not from filtering: the tool is
//! HIGH risk, so with the default threshold every invocation raises a
//! human approval card showing the exact command before anything runs.
#![forbid(unsafe_code)]

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, anyhow, bail};
use app_core::{RiskClass, SideEffect};
use async_trait::async_trait;
use serde_json::{Value, json};
use tool_core::{Tool, ToolContext, ToolDefinition, ToolResult};

/// Stdout+stderr kept per stream; the rest is summarized, not dropped
/// silently — the model sees that truncation happened.
const MAX_STREAM_CHARS: usize = 16 * 1024;
/// Agent-level backstop; per-call `timeout_ms` (<= 120 s) applies first.
const AGENT_TIMEOUT: Duration = Duration::from_secs(130);

pub fn process_tools() -> Vec<Arc<dyn Tool>> {
    vec![Arc::new(RunTool)]
}

fn run_definition() -> ToolDefinition {
    let mut def = ToolDefinition::new(
        "process.run",
        "Run a terminal command on this Windows PC (PowerShell by default, or cmd). \
         The command runs with the workspace root (or a subfolder of it) as working \
         directory. Returns the exit code plus stdout/stderr (truncated at 16 KB per \
         stream). Prefer non-interactive commands; the call fails after timeout_ms. \
         HIGH risk: every invocation requires human approval.",
        json!({
            "type": "object",
            "properties": {
                "shell": {
                    "type": "string",
                    "enum": ["powershell", "cmd"],
                    "description": "Shell to run under. Default 'powershell'."
                },
                "command": {
                    "type": "string",
                    "description": "The command line, e.g. 'cargo test --workspace' or 'dir'."
                },
                "workdir": {
                    "type": "string",
                    "description": "Subfolder of the workspace root to run in, or '.' for the root. Must stay inside the workspace."
                },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1000,
                    "maximum": 120000,
                    "description": "Kill the command after this long. Default 30000."
                }
            },
            "required": ["command"],
            "additionalProperties": false
        }),
        RiskClass::High,
        vec![SideEffect::RunsCommands],
    );
    def.workspace_scoped = true;
    def.timeout = AGENT_TIMEOUT;
    def
}

struct RunTool;

#[async_trait]
impl Tool for RunTool {
    fn definition(&self) -> &ToolDefinition {
        &RUN_DEF
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolResult> {
        let command = required_string(&args, "command")?;
        if command.trim().is_empty() {
            bail!("command is empty");
        }
        let shell = args
            .get("shell")
            .and_then(Value::as_str)
            .unwrap_or("powershell");
        let workdir = args.get("workdir").and_then(Value::as_str).unwrap_or(".");
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(Value::as_u64)
            .unwrap_or(30_000)
            .clamp(1_000, 120_000);

        let root = ctx.root().ok_or_else(|| {
            anyhow!("process.run needs a workspace: bind a workspace folder to the session first")
        })?;
        let cwd = resolve_under(root, workdir)?;

        let mut cmd = match shell {
            "powershell" => {
                let mut c = tokio::process::Command::new("powershell.exe");
                c.args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    &command,
                ]);
                c
            }
            "cmd" => {
                let mut c = tokio::process::Command::new("cmd.exe");
                c.args(["/d", "/c", &command]);
                c
            }
            other => bail!("unsupported shell '{other}': use 'powershell' or 'cmd'"),
        };
        cmd.current_dir(&cwd);
        // Non-interactive by construction: no stdin handle to hang on.
        cmd.stdin(std::process::Stdio::null());

        let output = tokio::time::timeout(Duration::from_millis(timeout_ms), cmd.output())
            .await
            .map_err(|_| anyhow!("command timed out after {timeout_ms} ms"))?
            .with_context(|| format!("failed to spawn {shell}"))?;

        let code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let (stdout, out_cut) = truncate(&stdout);
        let (stderr, err_cut) = truncate(&stderr);
        let truncated = out_cut || err_cut;

        let mut text = format!("exit: {code}\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}");
        if truncated {
            text.push_str("\n(truncated at 16 KB per stream)");
        }
        if truncated {
            Ok(ToolResult::truncated(text))
        } else {
            Ok(ToolResult::text(text))
        }
    }
}

fn truncate(s: &str) -> (String, bool) {
    if s.len() <= MAX_STREAM_CHARS {
        return (s.to_string(), false);
    }
    // Cut on a char boundary, never mid-codepoint.
    let mut end = MAX_STREAM_CHARS;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_string(), true)
}

fn required_string(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("missing required string argument '{key}'"))
}

/// Resolve `rel` inside `root`, rejecting absolute paths, `..` escapes and
/// symlinks that leave the root.
fn resolve_under(root: &Path, rel: &str) -> anyhow::Result<PathBuf> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        bail!("workdir must be relative to the workspace root");
    }
    if rel_path
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        // Fast reject before touching disk; canonicalization below is the
        // real check (covers symlinks).
        bail!("workdir must stay inside the workspace root");
    }
    let joined = root.join(rel_path);
    let canonical_root = root
        .canonicalize()
        .with_context(|| format!("workspace root does not exist: {}", root.display()))?;
    let canonical = joined
        .canonicalize()
        .with_context(|| format!("workdir does not exist: {}", joined.display()))?;
    if !canonical.starts_with(&canonical_root) {
        bail!("workdir must stay inside the workspace root");
    }
    Ok(canonical)
}

static RUN_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(run_definition);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tool_core::ToolContext;

    fn ctx_with(root: &Path) -> ToolContext {
        ToolContext {
            workspace: Some(app_core::WorkspaceConfig::from_dir(
                "test",
                root.to_path_buf(),
            )),
            cwd: Some(root.to_path_buf()),
        }
    }

    fn temp_root(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("super-ai-proc-test-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn run_tool() -> RunTool {
        RunTool
    }

    #[tokio::test]
    async fn powershell_echo_reports_exit_zero() {
        let root = temp_root("echo");
        let out = run_tool()
            .execute(
                json!({"command": "Write-Output hello-agent"}),
                &ctx_with(&root),
            )
            .await
            .unwrap();
        assert!(out.output.contains("exit: 0"), "got: {}", out.output);
        assert!(out.output.contains("hello-agent"), "got: {}", out.output);
        assert!(!out.truncated);
    }

    #[tokio::test]
    async fn cmd_shell_and_exit_code() {
        let root = temp_root("cmd");
        let out = run_tool()
            .execute(
                json!({"shell": "cmd", "command": "echo hi-cmd & exit 3"}),
                &ctx_with(&root),
            )
            .await
            .unwrap();
        assert!(out.output.contains("exit: 3"), "got: {}", out.output);
        assert!(out.output.contains("hi-cmd"), "got: {}", out.output);
    }

    #[tokio::test]
    async fn failing_command_reports_stderr() {
        let root = temp_root("fail");
        let out = run_tool()
            .execute(
                json!({"command": "Get-Item Z:\\definitely-not-here-xyz"}),
                &ctx_with(&root),
            )
            .await
            .unwrap();
        assert!(!out.output.contains("exit: 0"), "got: {}", out.output);
    }

    #[tokio::test]
    async fn timeout_kills_long_command() {
        let root = temp_root("timeout");
        let err = run_tool()
            .execute(
                json!({"command": "Start-Sleep -Seconds 30", "timeout_ms": 1500}),
                &ctx_with(&root),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("timed out"), "got: {err:#}");
    }

    #[tokio::test]
    async fn workdir_escape_rejected() {
        let root = temp_root("escape");
        for bad in ["..", "..\\..", "C:\\Windows"] {
            let err = run_tool()
                .execute(json!({"command": "dir", "workdir": bad}), &ctx_with(&root))
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("workspace"),
                "workdir {bad}: got {err:#}"
            );
        }
    }

    #[tokio::test]
    async fn bad_shell_and_empty_command_rejected() {
        let root = temp_root("args");
        let err = run_tool()
            .execute(json!({"shell": "bash", "command": "ls"}), &ctx_with(&root))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unsupported shell"));
        let err = run_tool()
            .execute(json!({"command": "   "}), &ctx_with(&root))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn missing_workspace_bails() {
        let err = run_tool()
            .execute(json!({"command": "dir"}), &ToolContext::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("workspace"));
    }

    #[test]
    fn definition_is_high_risk_and_scoped() {
        let def = run_definition();
        assert_eq!(def.name, "process.run");
        assert_eq!(def.risk, RiskClass::High);
        assert!(def.workspace_scoped);
        assert!(def.side_effects.contains(&SideEffect::RunsCommands));
    }
}
