use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::ModelVariant;

use super::JobStatus;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueResponse {
    pub id: Uuid,
    pub status: JobStatus,
    pub status_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub ready: bool,
    pub workers: WorkerHealth,
    pub queued: usize,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessResponse {
    pub status: &'static str,
    pub ready: bool,
    pub workers: WorkerHealth,
    pub queued: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerHealth {
    pub expected: usize,
    pub ready: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsResponse {
    pub workers: WorkerHealth,
    pub queued: usize,
    pub http_requests_total: usize,
    pub http_requests_failed: usize,
    pub total_request_ms: u128,
    pub average_request_ms: Option<f64>,
    pub jobs_started: usize,
    pub jobs_succeeded: usize,
    pub jobs_failed: usize,
    pub jobs_timed_out: usize,
    pub worker_restarts: usize,
    pub total_inference_ms: u128,
    pub average_inference_ms: Option<f64>,
    pub model_load_ms: Option<u128>,
    pub webhook_failures: usize,
    pub cleanup_failures: usize,
    pub retained_jobs: usize,
    pub process_memory_rss_bytes: Option<u64>,
    pub gpu_memory: Option<GpuMemoryResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryResponse {
    pub used_bytes: u64,
    pub total_bytes: u64,
}
