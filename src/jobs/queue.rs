use std::path::{Path, PathBuf};

use log::info;
use uuid::Uuid;

use crate::{
    state::AppState,
    types::{JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::append_jsonl,
};

#[derive(Debug, Clone)]
pub struct JobRequest {
    pub id: Uuid,
    pub image_path: PathBuf,
    pub task: TaskSpec,
}

#[derive(Debug, thiserror::Error)]
pub enum EnqueueError {
    #[error("inference queue is full")]
    QueueFull,
    #[error("inference queue is closed")]
    QueueClosed,
    #[error(transparent)]
    Persist(#[from] anyhow::Error),
}

pub async fn enqueue_record(
    state: &AppState,
    record: JobRecord,
) -> Result<QueueResponse, EnqueueError> {
    let id = record.id;
    let request = job_request_from_record(&record);

    if state.queue_tx.is_full() {
        return Err(EnqueueError::QueueFull);
    }

    {
        let mut jobs = state.jobs.write().await;
        jobs.insert(id, record.clone());
    }

    append_jsonl(&state.config.submissions_jsonl, &record)
        .await
        .map_err(EnqueueError::Persist)?;

    match state.queue_tx.try_send(request) {
        Ok(()) => {}
        Err(async_channel::TrySendError::Full(_)) => {
            rollback_unqueued_record(
                state,
                id,
                record,
                "inference queue became full before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueFull);
        }
        Err(async_channel::TrySendError::Closed(_)) => {
            rollback_unqueued_record(
                state,
                id,
                record,
                "inference queue closed before the job could be queued",
            )
            .await?;
            return Err(EnqueueError::QueueClosed);
        }
    }

    log_queued_job(&record, state.queue_tx.len());

    Ok(queue_response(id))
}

fn job_request_from_record(record: &JobRecord) -> JobRequest {
    JobRequest {
        id: record.id,
        image_path: record.image_path.clone(),
        task: TaskSpec::from_text_input(record.text_input.clone()),
    }
}

fn log_queued_job(record: &JobRecord, queued: usize) {
    info!(
        "job queued job_id={} text_input_present={} input_kind={} image_path={} image_bytes={} sha256={} queued={}",
        record.id,
        record.text_input.is_some(),
        record.input_kind,
        record.image_path.display(),
        record.image_bytes,
        record.image_sha256,
        queued
    );
}

fn queue_response(id: Uuid) -> QueueResponse {
    QueueResponse {
        id,
        status: JobStatus::Queued,
        status_url: format!("/v1/jobs/{id}"),
    }
}

async fn rollback_unqueued_record(
    state: &AppState,
    id: Uuid,
    record: JobRecord,
    error: &str,
) -> Result<(), EnqueueError> {
    state.jobs.write().await.remove(&id);
    persist_unqueued_record(&state.config.results_jsonl, record, error).await
}

async fn persist_unqueued_record(
    results_jsonl: &Path,
    mut record: JobRecord,
    error: &str,
) -> Result<(), EnqueueError> {
    record.status = JobStatus::Failed;
    record.updated_at = time::OffsetDateTime::now_utc();
    record.error = Some(error.to_string());
    append_jsonl(results_jsonl, &record)
        .await
        .map_err(EnqueueError::Persist)
}
