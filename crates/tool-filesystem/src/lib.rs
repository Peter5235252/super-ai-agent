//! Filesystem tools confined to a workspace root.
//!
//! Every path is resolved inside the root and canonicalized before use, so
//! `../` escapes and symlinks pointing outside the workspace are rejected.
//! This is the first line of defense; the sandbox broker (later phase)
//! enforces the same boundary at the OS level.
#![forbid(unsafe_code)]

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, anyhow, bail};
use app_core::{RiskClass, SideEffect};
use async_trait::async_trait;
use serde_json::{Value, json};
use tool_core::{Tool, ToolContext, ToolDefinition, ToolResult};

const MAX_READ_BYTES: usize = 200 * 1024;

pub fn filesystem_tools(root: PathBuf) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(ListTool { root: root.clone() }),
        Arc::new(ReadTool { root: root.clone() }),
        Arc::new(WriteTool { root }),
    ]
}

fn list_definition() -> ToolDefinition {
    ToolDefinition::new(
        "fs.list",
        "List the entries of a directory inside the workspace. Returns one line per entry: \
         name, size (bytes, - for directories), and a trailing '/' for directories.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path relative to the workspace root, or '.' for the root itself."}
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        RiskClass::Low,
        vec![SideEffect::ReadsFiles],
    )
    .with_workspace_scoped()
}

fn read_definition() -> ToolDefinition {
    ToolDefinition::new(
        "fs.read",
        "Read a text file inside the workspace. Output is truncated at 200 KB. Use for source \
         code, config, READMEs, and other text files.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."}
            },
            "required": ["path"],
            "additionalProperties": false
        }),
        RiskClass::Low,
        vec![SideEffect::ReadsFiles],
    )
    .with_workspace_scoped()
}

fn write_definition() -> ToolDefinition {
    ToolDefinition::new(
        "fs.write",
        "Create or overwrite a text file inside the workspace. Parent directories are created \
         automatically. MEDIUM risk: requires approval.",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace root."},
                "content": {"type": "string", "description": "Full new file contents."}
            },
            "required": ["path", "content"],
            "additionalProperties": false
        }),
        RiskClass::Medium,
        vec![SideEffect::WritesFiles],
    )
    .with_workspace_scoped()
}

trait WithWorkspaceScoped {
    fn with_workspace_scoped(self) -> Self;
}
impl WithWorkspaceScoped for ToolDefinition {
    fn with_workspace_scoped(mut self) -> Self {
        self.workspace_scoped = true;
        self
    }
}

struct ListTool {
    root: PathBuf,
}
struct ReadTool {
    root: PathBuf,
}
struct WriteTool {
    root: PathBuf,
}

#[async_trait]
impl Tool for ListTool {
    fn definition(&self) -> &ToolDefinition {
        &LIST_DEF
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolResult> {
        let path = required_string(&args, "path")?;
        let resolved = resolve(&self.root, &path, ctx)?;
        if !resolved.is_dir() {
            bail!("not a directory: {}", resolved.display());
        }
        let mut entries: Vec<String> = Vec::new();
        let mut read = tokio::fs::read_dir(&resolved)
            .await
            .with_context(|| format!("failed to list {}", resolved.display()))?;
        while let Some(entry) = read.next_entry().await? {
            let name = entry.file_name().to_string_lossy().to_string();
            let is_dir = entry.file_type().await?.is_dir();
            let size = if is_dir {
                "-".to_string()
            } else {
                entry
                    .metadata()
                    .await
                    .map(|m| m.len().to_string())
                    .unwrap_or_else(|_| "?".to_string())
            };
            entries.push(format!("{name}{} {size}", if is_dir { "/" } else { "" }));
        }
        entries.sort();
        Ok(ToolResult::text(entries.join("\n")))
    }
}

#[async_trait]
impl Tool for ReadTool {
    fn definition(&self) -> &ToolDefinition {
        &READ_DEF
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolResult> {
        let path = required_string(&args, "path")?;
        let resolved = resolve(&self.root, &path, ctx)?;
        if !resolved.is_file() {
            bail!("not a file: {}", resolved.display());
        }
        let bytes = tokio::fs::read(&resolved)
            .await
            .with_context(|| format!("failed to read {}", resolved.display()))?;
        if bytes.len() > MAX_READ_BYTES {
            let text = String::from_utf8_lossy(&bytes[..MAX_READ_BYTES]).to_string();
            return Ok(ToolResult::truncated(format!(
                "{text}\n\n[truncated: {} bytes, showing first {}]",
                bytes.len(),
                MAX_READ_BYTES
            )));
        }
        Ok(ToolResult::text(String::from_utf8_lossy(&bytes)))
    }
}

#[async_trait]
impl Tool for WriteTool {
    fn definition(&self) -> &ToolDefinition {
        &WRITE_DEF
    }

    async fn execute(&self, args: Value, ctx: &ToolContext) -> anyhow::Result<ToolResult> {
        let path = required_string(&args, "path")?;
        let content = args
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow!("missing string field 'content'"))?;
        let resolved = resolve(&self.root, &path, ctx)?;
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("failed to create parent dirs for {}", resolved.display())
            })?;
        }
        tokio::fs::write(&resolved, content)
            .await
            .with_context(|| format!("failed to write {}", resolved.display()))?;
        Ok(ToolResult::text(format!(
            "wrote {} bytes to {}",
            content.len(),
            resolved.display()
        )))
    }
}

fn required_string(args: &Value, key: &str) -> anyhow::Result<String> {
    args.get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| anyhow!("missing string field '{key}'"))
}

/// Resolve a user-supplied path inside `root`, rejecting escapes.
fn resolve(root: &Path, path: &str, _ctx: &ToolContext) -> anyhow::Result<PathBuf> {
    let root = root
        .canonicalize()
        .with_context(|| format!("workspace root does not exist: {}", root.display()))?;

    let joined = if path.is_empty() || path == "." {
        root.clone()
    } else {
        let candidate = Path::new(path);
        if candidate.is_absolute() {
            candidate.to_path_buf()
        } else {
            root.join(candidate)
        }
    };

    // Lexical check first (cheap, works before the file exists).
    if joined
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        bail!("path escapes the workspace root: '{path}'");
    }

    // Canonicalize the deepest existing ancestor so symlinks can't escape.
    let canonical = canonicalize_loose(&joined)?;
    if !is_within(&canonical, &root) {
        bail!("path escapes the workspace root: '{path}'");
    }
    Ok(canonical)
}

/// Canonicalize `p`; if `p` itself does not exist yet (e.g. for writes),
/// canonicalize the nearest existing ancestor and re-append the remainder.
fn canonicalize_loose(p: &Path) -> anyhow::Result<PathBuf> {
    if let Ok(c) = p.canonicalize() {
        return Ok(c);
    }
    let mut missing: Vec<PathBuf> = Vec::new();
    let mut ancestor = p.to_path_buf();
    loop {
        match ancestor.canonicalize() {
            Ok(c) => {
                let mut result = c;
                for part in missing.iter().rev() {
                    result.push(part);
                }
                return Ok(result);
            }
            Err(_) => {
                let name = ancestor
                    .file_name()
                    .ok_or_else(|| anyhow!("cannot resolve path: {}", p.display()))?;
                missing.push(PathBuf::from(name));
                if !ancestor.pop() {
                    bail!("cannot resolve path: {}", p.display());
                }
            }
        }
    }
}

fn is_within(path: &Path, root: &Path) -> bool {
    // Windows paths are case-insensitive.
    let path_str = path.to_string_lossy();
    let root_str = root.to_string_lossy();
    if cfg!(windows) {
        path_str
            .to_lowercase()
            .starts_with(&root_str.to_lowercase())
    } else {
        path_str.starts_with(root_str.as_ref())
    }
}

static LIST_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(list_definition);
static READ_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(read_definition);
static WRITE_DEF: std::sync::LazyLock<ToolDefinition> = std::sync::LazyLock::new(write_definition);

#[cfg(test)]
mod tests {
    use super::*;
    use app_core::WorkspaceConfig;
    use tool_core::ToolContext;

    fn ctx(root: &Path) -> ToolContext {
        ToolContext {
            workspace: Some(WorkspaceConfig::from_dir("test", root.to_path_buf())),
            cwd: Some(root.to_path_buf()),
        }
    }

    #[tokio::test]
    async fn write_read_list_roundtrip() {
        let dir = std::env::temp_dir().join(format!("super-ai-fs-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let tools = filesystem_tools(dir.clone());
        let ctx = ctx(&dir);

        let write = tools
            .iter()
            .find(|t| t.definition().name == "fs.write")
            .unwrap();
        let out = write
            .execute(
                json!({"path": "src/main.rs", "content": "fn main() {}"}),
                &ctx,
            )
            .await
            .unwrap();
        assert!(out.output.contains("wrote"));

        let read = tools
            .iter()
            .find(|t| t.definition().name == "fs.read")
            .unwrap();
        let out = read
            .execute(json!({"path": "src/main.rs"}), &ctx)
            .await
            .unwrap();
        assert_eq!(out.output, "fn main() {}");

        let list = tools
            .iter()
            .find(|t| t.definition().name == "fs.list")
            .unwrap();
        let out = list.execute(json!({"path": "."}), &ctx).await.unwrap();
        assert!(out.output.contains("src/"));

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn traversal_is_rejected() {
        let dir = std::env::temp_dir().join(format!("super-ai-fs-test-{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        let outside =
            std::env::temp_dir().join(format!("super-ai-outside-{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&outside, "secret").await.unwrap();

        let tools = filesystem_tools(dir.clone());
        let ctx = ctx(&dir);
        let read = tools
            .iter()
            .find(|t| t.definition().name == "fs.read")
            .unwrap();

        // Relative traversal: sibling of the root via `..`.
        let name = outside.file_name().unwrap().to_string_lossy().to_string();
        let path = format!("../{name}");
        assert!(read.execute(json!({"path": path}), &ctx).await.is_err());

        // Absolute path outside the root.
        let abs = outside.to_string_lossy().to_string();
        assert!(read.execute(json!({"path": abs}), &ctx).await.is_err());

        tokio::fs::remove_dir_all(&dir).await.unwrap();
        let _ = tokio::fs::remove_file(&outside).await;
    }

    #[test]
    fn definitions_are_typed() {
        let tools = filesystem_tools(std::env::temp_dir());
        let read = tools
            .iter()
            .find(|t| t.definition().name == "fs.read")
            .unwrap();
        assert_eq!(read.definition().risk, RiskClass::Low);
        assert!(read.definition().workspace_scoped);
        let write = tools
            .iter()
            .find(|t| t.definition().name == "fs.write")
            .unwrap();
        assert_eq!(write.definition().risk, RiskClass::Medium);
    }
}
