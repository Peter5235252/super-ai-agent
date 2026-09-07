//! SQLite persistence for sessions, messages, events, providers and
//! settings. Migrations are versioned from day one. Raw credentials are
//! never stored here — see the `secrets` crate.
#![forbid(unsafe_code)]

use app_core::{SessionId, TaskId, now_ms};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{FromRow, SqlitePool};
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum DbError {
    #[error("sqlx error: {0}")]
    Sqlx(#[from] sqlx::Error),
    #[error("migration error: {0}")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("bad uuid '{0}': {1}")]
    BadUuid(String, #[source] uuid::Error),
}

pub type Result<T> = std::result::Result<T, DbError>;

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SessionRow {
    pub id: Uuid,
    pub title: String,
    pub workspace: Option<String>,
    pub provider_name: Option<String>,
    pub model: Option<String>,
    pub status: String,
    /// Reasoning effort override (off/low/medium/high/max); None = Default.
    pub reasoning_effort: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MessageRow {
    pub id: Uuid,
    pub session_id: Uuid,
    pub role: String,
    pub content: String,
    pub tool_calls: Option<String>,
    pub tool_call_id: Option<String>,
    /// The model's private reasoning for assistant turns (UI only).
    pub reasoning: Option<String>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct EventRow {
    pub id: i64,
    pub task_id: Option<String>,
    pub session_id: Option<String>,
    pub kind: String,
    pub payload: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct ProviderRow {
    pub name: String,
    pub kind: String,
    pub base_url: Option<String>,
    pub default_model: Option<String>,
    pub is_default: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone)]
pub struct Db {
    pool: SqlitePool,
}

impl Db {
    pub async fn open(path: &Path) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(path.to_str().ok_or_else(|| {
            DbError::Sqlx(sqlx::Error::Configuration(
                "database path is not valid UTF-8".into(),
            ))
        })?)?
        .create_if_missing(true)
        .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(options)
            .await?;
        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    #[cfg(test)]
    pub async fn open_in_memory() -> Result<Self> {
        let options = SqliteConnectOptions::from_str("sqlite::memory:")?.foreign_keys(true);
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;
        let db = Self { pool };
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> Result<()> {
        let migrator =
            sqlx::migrate::Migrator::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("migrations"))
                .await?;
        migrator.run(&self.pool).await?;
        Ok(())
    }

    // ---- sessions ----

    pub async fn create_session(
        &self,
        id: Uuid,
        title: &str,
        workspace: Option<&str>,
    ) -> Result<SessionRow> {
        let now = now_ms();
        sqlx::query_as::<_, SessionRow>(
            "INSERT INTO sessions (id, title, workspace, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             RETURNING *",
        )
        .bind(id)
        .bind(title)
        .bind(workspace)
        .bind(now)
        .fetch_one(&self.pool)
        .await
        .map_err(Into::into)
    }

    pub async fn get_session(&self, id: SessionId) -> Result<Option<SessionRow>> {
        sqlx::query_as::<_, SessionRow>("SELECT * FROM sessions WHERE id = ?1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionRow>> {
        sqlx::query_as::<_, SessionRow>("SELECT * FROM sessions ORDER BY updated_at DESC")
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn touch_session(&self, id: SessionId) -> Result<()> {
        sqlx::query("UPDATE sessions SET updated_at = ?1 WHERE id = ?2")
            .bind(now_ms())
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn bind_provider(
        &self,
        session: SessionId,
        provider: &str,
        model: &str,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE sessions SET provider_name = ?1, model = ?2, updated_at = ?3 WHERE id = ?4",
        )
        .bind(provider)
        .bind(model)
        .bind(now_ms())
        .bind(session)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn set_session_effort(
        &self,
        session: SessionId,
        effort: Option<&str>,
    ) -> Result<()> {
        sqlx::query("UPDATE sessions SET reasoning_effort = ?1, updated_at = ?2 WHERE id = ?3")
            .bind(effort)
            .bind(now_ms())
            .bind(session)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn delete_session(&self, id: SessionId) -> Result<()> {
        sqlx::query("DELETE FROM sessions WHERE id = ?1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ---- messages ----

    pub async fn insert_message(&self, row: &MessageRow) -> Result<()> {
        sqlx::query(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, tool_call_id, reasoning, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )
        .bind(row.id)
        .bind(row.session_id)
        .bind(&row.role)
        .bind(&row.content)
        .bind(&row.tool_calls)
        .bind(&row.tool_call_id)
        .bind(&row.reasoning)
        .bind(row.created_at)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_messages(&self, session: SessionId) -> Result<Vec<MessageRow>> {
        sqlx::query_as::<_, MessageRow>(
            "SELECT * FROM messages WHERE session_id = ?1 ORDER BY created_at, rowid",
        )
        .bind(session)
        .fetch_all(&self.pool)
        .await
        .map_err(Into::into)
    }

    // ---- events (flight recorder) ----

    pub async fn insert_event(
        &self,
        task_id: Option<TaskId>,
        session_id: Option<SessionId>,
        kind: &str,
        payload: &Value,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO events (task_id, session_id, kind, payload, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
        )
        .bind(task_id.map(|t| t.to_string()))
        .bind(session_id.map(|s| s.to_string()))
        .bind(kind)
        .bind(serde_json::to_string(payload).unwrap_or_else(|_| "{}".into()))
        .bind(now_ms())
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_events(&self, session: SessionId, limit: i64) -> Result<Vec<EventRow>> {
        sqlx::query_as::<_, EventRow>(
            "SELECT * FROM events WHERE session_id = ?1 ORDER BY id DESC LIMIT ?2",
        )
        .bind(session.to_string())
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map_err(Into::into)
    }

    // ---- providers ----

    pub async fn upsert_provider(&self, row: &ProviderRow) -> Result<()> {
        let now = now_ms();
        sqlx::query(
            "INSERT INTO providers (name, kind, base_url, default_model, is_default, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(name) DO UPDATE SET
                kind = ?2, base_url = ?3, default_model = ?4, updated_at = ?7",
        )
        .bind(&row.name)
        .bind(&row.kind)
        .bind(&row.base_url)
        .bind(&row.default_model)
        .bind(row.is_default)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_providers(&self) -> Result<Vec<ProviderRow>> {
        sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers ORDER BY name")
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn get_provider(&self, name: &str) -> Result<Option<ProviderRow>> {
        sqlx::query_as::<_, ProviderRow>("SELECT * FROM providers WHERE name = ?1")
            .bind(name)
            .fetch_optional(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub async fn delete_provider(&self, name: &str) -> Result<()> {
        sqlx::query("DELETE FROM providers WHERE name = ?1")
            .bind(name)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // ---- settings ----

    pub async fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let row: Option<(String,)> = sqlx::query_as("SELECT value FROM settings WHERE key = ?1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.0))
    }

    pub async fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = ?2",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn all_settings(&self) -> Result<Vec<(String, String)>> {
        sqlx::query_as("SELECT key, value FROM settings ORDER BY key")
            .fetch_all(&self.pool)
            .await
            .map_err(Into::into)
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn session_message_roundtrip() {
        let db = Db::open_in_memory().await.unwrap();
        let session_id = Uuid::new_v4();
        let s = db
            .create_session(session_id, "Test", Some("C:\\work"))
            .await
            .unwrap();
        assert_eq!(s.title, "Test");
        assert_eq!(s.workspace.as_deref(), Some("C:\\work"));

        db.insert_message(&MessageRow {
            id: Uuid::new_v4(),
            session_id,
            role: "user".into(),
            content: "hello".into(),
            tool_calls: None,
            tool_call_id: None,
            reasoning: None,
            created_at: now_ms(),
        })
        .await
        .unwrap();

        let messages = db.list_messages(session_id).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "hello");

        let sessions = db.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
    }

    #[tokio::test]
    async fn tool_turn_roundtrip_keeps_pairing() {
        // A full agent turn (assistant tool_calls + tool results + reasoning)
        // must reload intact, or providers reject the next request.
        let db = Db::open_in_memory().await.unwrap();
        let session_id = Uuid::new_v4();
        db.create_session(session_id, "Tools", None).await.unwrap();

        let mk = |role: &str,
                  content: &str,
                  tool_calls: Option<String>,
                  tool_call_id: Option<String>,
                  reasoning: Option<String>| MessageRow {
            id: Uuid::new_v4(),
            session_id,
            role: role.into(),
            content: content.into(),
            tool_calls,
            tool_call_id,
            reasoning,
            created_at: now_ms(),
        };
        db.insert_message(&mk("user", "list files", None, None, None))
            .await
            .unwrap();
        db.insert_message(&mk(
            "assistant",
            "",
            Some(r#"[{"id":"call_1","name":"fs.list","arguments":{"path":"."}}]"#.into()),
            None,
            Some("need file list first".into()),
        ))
        .await
        .unwrap();
        db.insert_message(&mk("tool", "a.txt", None, Some("call_1".into()), None))
            .await
            .unwrap();

        let messages = db.list_messages(session_id).await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, "user");
        let assistant = &messages[1];
        assert_eq!(assistant.role, "assistant");
        assert!(assistant.tool_calls.as_deref().unwrap().contains("fs.list"));
        assert_eq!(
            assistant.reasoning.as_deref(),
            Some("need file list first")
        );
        let tool = &messages[2];
        assert_eq!(tool.role, "tool");
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn provider_and_settings_roundtrip() {
        let db = Db::open_in_memory().await.unwrap();
        db.upsert_provider(&ProviderRow {
            name: "my-openai".into(),
            kind: "openai".into(),
            base_url: None,
            default_model: Some("gpt-6-astra".into()),
            is_default: true,
            created_at: now_ms(),
            updated_at: now_ms(),
        })
        .await
        .unwrap();
        let p = db.get_provider("my-openai").await.unwrap().unwrap();
        assert_eq!(p.default_model.as_deref(), Some("gpt-6-astra"));

        db.set_setting("auto_approve_risk", "low").await.unwrap();
        assert_eq!(
            db.get_setting("auto_approve_risk")
                .await
                .unwrap()
                .as_deref(),
            Some("low")
        );
        assert_eq!(db.all_settings().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn events_persist_payloads() {
        let db = Db::open_in_memory().await.unwrap();
        let session = Uuid::new_v4();
        let task = Uuid::new_v4();
        db.insert_event(
            Some(task),
            Some(session),
            "task_created",
            &serde_json::json!({"model": "grok-4.6"}),
        )
        .await
        .unwrap();
        let events = db.list_events(session, 10).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, "task_created");
        assert!(events[0].payload.contains("grok-4.6"));
    }
}
