mod validation;

use std::path::PathBuf;

use anyhow::Context;
use axum::{
    Json,
    body::Bytes,
    extract::{Multipart, State, multipart::Field},
    response::Html,
};
use log::{debug, trace, warn};
use serde::Deserialize;
use time::OffsetDateTime;
use uuid::Uuid;

use crate::{
    config::Config,
    jobs::{EnqueueError, enqueue_record},
    state::AppState,
    types::{JobRecord, JobStatus, QueueResponse, TaskSpec},
    util::{guess_extension, image_format_content_type, sha256_hex},
};

pub(super) use self::validation::{
    ensure_local_path_allowed, ensure_workers_ready, validate_image_bytes, validate_webhook_url,
};
use super::{ApiError, render_index};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct LocalPathRequest {
    image_path: PathBuf,
    text_input: Option<String>,
    webhook_url: Option<String>,
}

struct UploadedImage {
    filename: Option<String>,
    content_type: Option<String>,
    bytes: Bytes,
}

#[derive(Default)]
struct MultipartInferenceRequest {
    image: Option<UploadedImage>,
    text_input: Option<String>,
    webhook_url: Option<String>,
}

pub(super) async fn submit_inference(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<QueueResponse>, ApiError> {
    Ok(Json(submit_multipart_job(&state, multipart).await?))
}

pub(super) async fn submit_inference_form(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Html<String>, ApiError> {
    match submit_multipart_job(&state, multipart).await {
        Ok(response) => render_index(Some(response), None),
        Err(err) => render_index(None, Some(err.to_string())),
    }
}

async fn submit_multipart_job(
    state: &AppState,
    mut multipart: Multipart,
) -> Result<QueueResponse, ApiError> {
    debug!("multipart inference submission started");
    ensure_workers_ready(state)?;
    let request = read_multipart_inference_request(&mut multipart).await?;
    let record = uploaded_job_record(state, request).await?;

    enqueue_record(state, record).await.map_err(ApiError::from)
}

async fn read_multipart_inference_request(
    multipart: &mut Multipart,
) -> Result<MultipartInferenceRequest, ApiError> {
    let mut request = MultipartInferenceRequest::default();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read multipart field: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        trace!("multipart field received name={}", name);
        apply_multipart_field(&mut request, name.as_str(), field).await?;
    }

    Ok(request)
}

async fn apply_multipart_field(
    request: &mut MultipartInferenceRequest,
    name: &str,
    field: Field<'_>,
) -> Result<(), ApiError> {
    match name {
        "image" | "file" => request.image = Some(read_uploaded_image(field).await?),
        "task_type" | "task_type_selector" | "task_prompt" | "task" => {
            return Err(unsupported_prompt_field(name));
        }
        "text_input" => request.text_input = Some(read_text_field(field, "text_input").await?),
        "webhook_url" => request.webhook_url = Some(read_text_field(field, "webhook_url").await?),
        _ => {}
    }

    Ok(())
}

fn unsupported_prompt_field(name: &str) -> ApiError {
    ApiError::BadRequest(format!(
        "multipart field `{name}` is not supported; use `text_input` for prompt overrides"
    ))
}

async fn read_uploaded_image(field: Field<'_>) -> Result<UploadedImage, ApiError> {
    let filename = field.file_name().map(str::to_string);
    let content_type = field.content_type().map(str::to_string);
    let bytes = field
        .bytes()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read image field: {err}")))?;
    debug!(
        "image field read filename={:?} content_type={:?} bytes={}",
        filename,
        content_type,
        bytes.len()
    );
    Ok(UploadedImage {
        filename,
        content_type,
        bytes,
    })
}

async fn read_text_field(field: Field<'_>, field_name: &str) -> Result<String, ApiError> {
    field
        .text()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read {field_name} field: {err}")))
}

async fn uploaded_job_record(
    state: &AppState,
    request: MultipartInferenceRequest,
) -> Result<JobRecord, ApiError> {
    let task = task_spec_from_request(request.text_input);
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    let image = request
        .image
        .ok_or_else(|| ApiError::BadRequest("multipart field `image` is required".into()))?;
    if image.bytes.is_empty() {
        warn!("rejecting empty image upload");
        return Err(ApiError::BadRequest("uploaded image is empty".into()));
    }
    validate_image_bytes(state, image.content_type.as_deref(), &image.bytes)?;

    let id = Uuid::new_v4();
    let now = OffsetDateTime::now_utc();
    let image_path = save_uploaded_image(&state.config, id, &image).await?;
    let image_sha256 = sha256_hex(&image.bytes);
    Ok(JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        image_path: image_path.clone(),
        filename: image.filename,
        content_type: image.content_type,
        image_sha256,
        image_bytes: image.bytes.len(),
        input_kind: "upload".to_string(),
        source_path: None,
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    })
}

async fn save_uploaded_image(
    config: &Config,
    id: Uuid,
    image: &UploadedImage,
) -> Result<PathBuf, ApiError> {
    let extension = guess_extension(image.content_type.as_deref(), &image.bytes);
    let image_path = config.images_dir.join(format!("{id}.{extension}"));
    tokio::fs::write(&image_path, &image.bytes)
        .await
        .with_context(|| format!("failed to save uploaded image to {}", image_path.display()))?;
    debug!(
        "uploaded image saved job_id={} path={} bytes={} extension={}",
        id,
        image_path.display(),
        image.bytes.len(),
        extension
    );
    Ok(image_path)
}

pub(super) async fn submit_inference_path(
    State(state): State<AppState>,
    Json(request): Json<LocalPathRequest>,
) -> Result<Json<QueueResponse>, ApiError> {
    ensure_workers_ready(&state)?;
    debug!(
        "local path inference submission started image_path={}",
        request.image_path.display()
    );

    let record = local_path_job_record(&state, request).await?;

    Ok(Json(enqueue_record(&state, record).await?))
}

async fn local_path_job_record(
    state: &AppState,
    request: LocalPathRequest,
) -> Result<JobRecord, ApiError> {
    let image_path = validated_local_image_path(state, &request.image_path).await?;
    let bytes = read_local_image(&image_path).await?;
    let content_type = image::guess_format(&bytes)
        .ok()
        .and_then(image_format_content_type)
        .map(str::to_string);
    validate_image_bytes(state, content_type.as_deref(), &bytes)?;

    let id = Uuid::new_v4();
    let task = task_spec_from_request(request.text_input);
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    let filename = image_path
        .file_name()
        .map(|filename| filename.to_string_lossy().into_owned());

    debug!(
        "local image accepted job_id={} path={} bytes={} content_type={:?}",
        id,
        image_path.display(),
        bytes.len(),
        content_type
    );

    let now = OffsetDateTime::now_utc();
    Ok(JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        image_path: image_path.clone(),
        filename,
        content_type,
        image_sha256: sha256_hex(&bytes),
        image_bytes: bytes.len(),
        input_kind: "local_path".to_string(),
        source_path: Some(image_path),
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    })
}

async fn validated_local_image_path(
    state: &AppState,
    requested_path: &PathBuf,
) -> Result<PathBuf, ApiError> {
    if requested_path.as_os_str().is_empty() {
        return Err(ApiError::BadRequest("`image_path` is required".into()));
    }

    let image_path = tokio::fs::canonicalize(requested_path)
        .await
        .map_err(|err| {
            ApiError::BadRequest(format!(
                "failed to resolve image path {}: {err}",
                requested_path.display()
            ))
        })?;
    ensure_local_path_allowed(state, &image_path).await?;
    let metadata = tokio::fs::metadata(&image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to inspect image path {}: {err}",
            image_path.display()
        ))
    })?;
    if !metadata.is_file() {
        return Err(ApiError::BadRequest(format!(
            "image path is not a regular file: {}",
            image_path.display()
        )));
    }
    Ok(image_path)
}

async fn read_local_image(image_path: &PathBuf) -> Result<Vec<u8>, ApiError> {
    let bytes = tokio::fs::read(image_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to read image path {}: {err}",
            image_path.display()
        ))
    })?;
    if bytes.is_empty() {
        warn!("rejecting empty local image path={}", image_path.display());
        return Err(ApiError::BadRequest("local image file is empty".into()));
    }
    Ok(bytes)
}

impl From<EnqueueError> for ApiError {
    fn from(err: EnqueueError) -> Self {
        match err {
            EnqueueError::QueueFull => {
                ApiError::ServiceUnavailable("inference queue is full".to_string())
            }
            EnqueueError::QueueClosed => {
                ApiError::ServiceUnavailable("inference queue is closed".to_string())
            }
            EnqueueError::Persist(source) => ApiError::Internal(source),
        }
    }
}

pub(super) fn task_spec_from_request(text_input: Option<String>) -> TaskSpec {
    TaskSpec::from_text_input(text_input)
}
