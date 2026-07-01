mod docs;
mod security;
mod submit;
mod system;

use std::time::{Duration, Instant};

use askama::Template;
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, Request, State},
    http::StatusCode,
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use log::{debug, info};
use serde_json::Value;
use tower_http::timeout::TimeoutLayer;
use uuid::Uuid;

use crate::{
    state::AppState,
    templates::{
        IndexTemplate, JobHtmlDetectionView, JobHtmlTableCellView, JobHtmlTableView,
        JobHtmlTemplate,
    },
    types::SubmissionResponse,
    types::{
        ErrorResponse, HealthResponse, JobRecord, MetricsResponse, ReadinessResponse, WorkerHealth,
    },
    types::{JobStatus, OcrDetection, OcrResult, OcrTable},
};

use self::{
    docs::{cors_layer, openapi_document},
    security::{add_security_headers, rate_limit, require_api_key},
    submit::{submit_inference, submit_inference_form, submit_inference_path},
    system::system_usage,
};

pub fn router(state: AppState) -> Router {
    let cors_allowed_origins = state.config.cors_allowed_origins.clone();
    let request_timeout_seconds = state.config.request_timeout_seconds;
    let state_for_middleware = state.clone();
    let mut router = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/ready", get(readiness))
        .route("/metrics", get(metrics))
        .route("/openapi.json", get(openapi))
        .route("/v1/infer", post(submit_inference))
        .route("/infer-form", post(submit_inference_form))
        .route("/v1/infer/path", post(submit_inference_path))
        .route("/v1/jobs/{id}", get(get_job))
        .route("/v1/jobs/{id}/html", get(get_job_html))
        .layer(DefaultBodyLimit::max(state.config.body_limit_bytes))
        .layer(middleware::from_fn_with_state(
            state_for_middleware.clone(),
            rate_limit,
        ))
        .layer(middleware::from_fn_with_state(
            state_for_middleware.clone(),
            require_api_key,
        ))
        .with_state(state);

    if request_timeout_seconds > 0 {
        router = router.layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_secs(request_timeout_seconds),
        ));
    }

    let router = if cors_allowed_origins.is_empty() {
        router
    } else {
        router.layer(cors_layer(&cors_allowed_origins))
    };

    router
        .layer(middleware::from_fn_with_state(
            state_for_middleware,
            track_request_metrics,
        ))
        .layer(middleware::from_fn(add_security_headers))
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("job not found")]
    NotFound,
    #[error("{0}")]
    Forbidden(String),
    #[error("{0}")]
    ServiceUnavailable(String),
    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::NotFound => StatusCode::NOT_FOUND,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let body = Json(ErrorResponse {
            code: self.code().to_string(),
            message: self.to_string(),
        });
        (status, body).into_response()
    }
}

impl ApiError {
    fn code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::NotFound => "not_found",
            ApiError::Forbidden(_) => "forbidden",
            ApiError::ServiceUnavailable(_) => "service_unavailable",
            ApiError::Internal(_) => "internal_error",
        }
    }
}

async fn index() -> Result<Html<String>, ApiError> {
    render_index(None, None)
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let workers = worker_health(&state);
    debug!(
        "health check worker_ready={} workers={} ready_workers={} failed_workers={} queued={} model_path={} model_variant={}",
        workers.ready > 0,
        workers.expected,
        workers.ready,
        workers.failed,
        state.queue_tx.len(),
        state.config.model_path.display(),
        state.config.model_variant.as_str()
    );

    Json(HealthResponse {
        status: "ok",
        ready: workers.ready > 0,
        workers,
        queued: state.queue_tx.len(),
        model_path: state.config.model_path.clone(),
        model_variant: state.config.model_variant,
    })
}

async fn readiness(State(state): State<AppState>) -> Response {
    let workers = worker_health(&state);
    let ready = workers.ready > 0;
    let response = ReadinessResponse {
        status: if ready { "ready" } else { "not_ready" },
        ready,
        workers,
        queued: state.queue_tx.len(),
    };
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (status, Json(response)).into_response()
}

async fn metrics(State(state): State<AppState>) -> Json<MetricsResponse> {
    let workers = worker_health(&state);
    let metrics = state.metrics.snapshot();
    let completed = metrics.jobs_succeeded + metrics.jobs_failed;
    let average_inference_ms =
        (completed > 0).then_some(metrics.total_inference_ms as f64 / completed as f64);
    let average_request_ms = (metrics.http_requests_total > 0)
        .then_some(metrics.total_request_ms as f64 / metrics.http_requests_total as f64);
    let (process_memory_rss_bytes, gpu_memory) = system_usage().await;

    Json(MetricsResponse {
        workers,
        queued: state.queue_tx.len(),
        http_requests_total: metrics.http_requests_total,
        http_requests_failed: metrics.http_requests_failed,
        total_request_ms: metrics.total_request_ms as u128,
        average_request_ms,
        jobs_started: metrics.jobs_started,
        jobs_succeeded: metrics.jobs_succeeded,
        jobs_failed: metrics.jobs_failed,
        jobs_timed_out: metrics.jobs_timed_out,
        worker_restarts: metrics.worker_restarts,
        total_inference_ms: metrics.total_inference_ms as u128,
        average_inference_ms,
        model_load_ms: metrics.model_load_ms.map(|value| value as u128),
        webhook_failures: metrics.webhook_failures,
        cleanup_failures: metrics.cleanup_failures,
        retained_jobs: state.jobs.read().await.len(),
        process_memory_rss_bytes,
        gpu_memory,
    })
}

async fn openapi() -> Json<Value> {
    Json(openapi_document())
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Json<JobRecord>, ApiError> {
    let jobs = state.jobs.read().await;
    let record = jobs.get(&id).cloned().ok_or(ApiError::NotFound)?;
    debug!("job status read job_id={} status={:?}", id, record.status);
    Ok(Json(record))
}

async fn get_job_html(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<Uuid>,
) -> Result<Html<String>, ApiError> {
    let jobs = state.jobs.read().await;
    let record = jobs.get(&id).cloned().ok_or(ApiError::NotFound)?;
    debug!("job html read job_id={} status={:?}", id, record.status);
    render_job_html(&record)
}

async fn track_request_metrics(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let method = request.method().clone();
    let uri = request.uri().clone();
    let version = request.version();
    let content_length = request
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<usize>().ok());
    let started = Instant::now();

    debug!(
        "http request started method={} uri={} version={:?} content_length={:?}",
        method, uri, version, content_length
    );

    let response = next.run(request).await;
    let elapsed = started.elapsed();
    let failed = response.status().is_client_error() || response.status().is_server_error();
    state
        .metrics
        .record_http_request(elapsed.as_millis(), failed);
    info!(
        "http request finished method={} uri={} status={} elapsed_ms={}",
        method,
        uri,
        response.status(),
        elapsed.as_millis()
    );

    response
}

fn render_job_html(record: &JobRecord) -> Result<Html<String>, ApiError> {
    let result = record.result.as_ref();
    let parsed = result.and_then(|metadata| {
        serde_json::from_value::<OcrResult>(metadata.result.clone())
            .map_err(|err| {
                debug!("failed to parse structured OCR result for html view error={err}");
                err
            })
            .ok()
    });
    let detections = parsed
        .as_ref()
        .map(job_html_detection_views)
        .unwrap_or_default();
    let raw_text = parsed
        .as_ref()
        .map(|result| result.text.clone())
        .or_else(|| result.map(|metadata| metadata.generated_text.clone()))
        .unwrap_or_default();
    let prompt_text = result
        .map(|metadata| metadata.prompt_text.clone())
        .unwrap_or_default();
    let generated_tokens = result
        .map(|metadata| metadata.generated_tokens)
        .unwrap_or_default();
    let elapsed_ms = result
        .map(|metadata| metadata.elapsed_ms)
        .unwrap_or_default();
    let has_detections = !detections.is_empty();

    let html = JobHtmlTemplate {
        id: record.id.to_string(),
        status: job_status_label(&record.status).to_string(),
        updated_at: record.updated_at.to_string(),
        filename: job_input_label(record),
        has_result: result.is_some(),
        error: record.error.clone().unwrap_or_default(),
        prompt_text,
        generated_tokens,
        elapsed_ms,
        detections,
        has_detections,
        raw_text,
    }
    .render()
    .map_err(|err| anyhow::anyhow!("failed to render job html view: {err}"))?;

    Ok(Html(html))
}

fn job_html_detection_views(result: &OcrResult) -> Vec<JobHtmlDetectionView> {
    let mut used_tables = vec![false; result.tables.len()];
    result
        .detections
        .iter()
        .map(|detection| {
            let tables = matching_table_views(detection, &result.tables, &mut used_tables);
            let has_tables = !tables.is_empty();
            JobHtmlDetectionView {
                label: detection.label.clone(),
                bbox: format_bbox(&detection.bbox),
                text: detection.text.clone(),
                tables,
                has_tables,
            }
        })
        .collect()
}

fn matching_table_views(
    detection: &OcrDetection,
    tables: &[OcrTable],
    used_tables: &mut [bool],
) -> Vec<JobHtmlTableView> {
    if !detection.label.eq_ignore_ascii_case("table") {
        return Vec::new();
    }

    let Some(index) = tables.iter().enumerate().position(|(index, table)| {
        !used_tables[index] && table.bbox == detection.bbox && table.html == detection.text
    }) else {
        return Vec::new();
    };
    used_tables[index] = true;

    vec![JobHtmlTableView {
        rows: tables[index]
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| JobHtmlTableCellView {
                        text: cell.text.clone(),
                        row_span: cell.row_span,
                        col_span: cell.col_span,
                        has_row_span: cell.row_span > 1,
                        has_col_span: cell.col_span > 1,
                    })
                    .collect()
            })
            .collect(),
    }]
}

fn format_bbox(bbox: &crate::types::BoundingBox) -> String {
    format!(
        "[{}, {}, {}, {}]",
        bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max
    )
}

fn job_status_label(status: &JobStatus) -> &'static str {
    match status {
        JobStatus::Queued => "queued",
        JobStatus::Running => "running",
        JobStatus::Succeeded => "succeeded",
        JobStatus::Failed => "failed",
    }
}

fn job_input_label(record: &JobRecord) -> String {
    record
        .filename
        .clone()
        .or_else(|| record.document_filename.clone())
        .unwrap_or_else(|| record.image_path.display().to_string())
}

fn render_index(
    response: Option<SubmissionResponse>,
    error: Option<String>,
) -> Result<Html<String>, ApiError> {
    let (queued_message, status_url) = queued_submission_view(response.as_ref());
    let template = IndexTemplate {
        queued: response.is_some(),
        queued_message,
        status_url,
        has_status_url: response.is_some(),
        error: error.unwrap_or_default(),
    };

    template.render().map(Html).map_err(|err| {
        ApiError::Internal(anyhow::anyhow!("failed to render index template: {err}"))
    })
}

fn queued_submission_view(response: Option<&SubmissionResponse>) -> (String, String) {
    match response {
        Some(SubmissionResponse::Image(response)) => (
            format!("Job queued: {}", response.id),
            response.status_url.clone(),
        ),
        Some(SubmissionResponse::Pdf(response)) => {
            let first_status_url = response
                .jobs
                .first()
                .map(|job| job.status_url.clone())
                .unwrap_or_default();
            (
                format!("PDF queued: {} page jobs", response.page_count),
                first_status_url,
            )
        }
        None => (String::new(), String::new()),
    }
}

fn worker_health(state: &AppState) -> WorkerHealth {
    let snapshot = state.workers.snapshot();
    WorkerHealth {
        expected: snapshot.expected,
        ready: snapshot.ready,
        failed: snapshot.failed,
    }
}

#[cfg(test)]
mod tests;
