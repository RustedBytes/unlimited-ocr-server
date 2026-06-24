use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use super::InferenceMetadata;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRecord {
    pub id: Uuid,
    pub status: JobStatus,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: OffsetDateTime,
    #[serde(with = "time::serde::rfc3339")]
    pub updated_at: OffsetDateTime,
    pub image_path: PathBuf,
    pub filename: Option<String>,
    pub content_type: Option<String>,
    pub image_sha256: String,
    pub image_bytes: usize,
    pub input_kind: String,
    pub source_path: Option<PathBuf>,
    pub task_type: String,
    pub task_prompt: String,
    pub text_input: Option<String>,
    pub webhook_url: Option<String>,
    pub result: Option<InferenceMetadata>,
    pub error: Option<String>,
}
