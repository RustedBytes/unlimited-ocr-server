mod validation;

use std::path::PathBuf;

use anyhow::{Context, anyhow};
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

use crate::pdf::{
    PdfRenderError, PdfRenderOptions, RenderedPdfPage, is_pdf_content, render_pdf_pages,
};
use crate::{
    config::Config,
    jobs::{EnqueueError, enqueue_record},
    state::AppState,
    types::{JobRecord, JobStatus, PdfQueueResponse, SubmissionResponse, TaskSpec},
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

struct UploadedFile {
    filename: Option<String>,
    content_type: Option<String>,
    bytes: Bytes,
}

#[derive(Default)]
struct MultipartInferenceRequest {
    file: Option<UploadedFile>,
    text_input: Option<String>,
    webhook_url: Option<String>,
}

pub(super) async fn submit_inference(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Json<SubmissionResponse>, ApiError> {
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
) -> Result<SubmissionResponse, ApiError> {
    debug!("multipart inference submission started");
    ensure_workers_ready(state)?;
    let request = read_multipart_inference_request(&mut multipart).await?;
    let submission = uploaded_pending_submission(state, request).await?;

    enqueue_pending_submission(state, submission).await
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
        "image" | "file" => request.file = Some(read_uploaded_file(field).await?),
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

async fn read_uploaded_file(field: Field<'_>) -> Result<UploadedFile, ApiError> {
    let filename = field.file_name().map(str::to_string);
    let content_type = field.content_type().map(str::to_string);
    let bytes = field
        .bytes()
        .await
        .map_err(|err| ApiError::BadRequest(format!("failed to read file field: {err}")))?;
    debug!(
        "file field read filename={:?} content_type={:?} bytes={}",
        filename,
        content_type,
        bytes.len()
    );
    Ok(UploadedFile {
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

enum PendingSubmission {
    Image(Box<JobRecord>),
    Pdf(Vec<JobRecord>),
}

async fn enqueue_pending_submission(
    state: &AppState,
    submission: PendingSubmission,
) -> Result<SubmissionResponse, ApiError> {
    match submission {
        PendingSubmission::Image(record) => enqueue_record(state, *record)
            .await
            .map(SubmissionResponse::Image)
            .map_err(ApiError::from),
        PendingSubmission::Pdf(records) => {
            if let Err(err) = ensure_queue_has_capacity(state, records.len()) {
                cleanup_rendered_pages(&records).await;
                return Err(err);
            }
            enqueue_pdf_records(state, records).await
        }
    }
}

async fn enqueue_pdf_records(
    state: &AppState,
    records: Vec<JobRecord>,
) -> Result<SubmissionResponse, ApiError> {
    let page_count = records.len();
    let mut responses = Vec::with_capacity(page_count);
    for record in records {
        let response = enqueue_record(state, record)
            .await
            .map_err(ApiError::from)?;
        responses.push(response);
    }

    Ok(SubmissionResponse::Pdf(PdfQueueResponse {
        kind: "pdf".to_string(),
        page_count,
        jobs: responses,
    }))
}

fn ensure_queue_has_capacity(state: &AppState, jobs: usize) -> Result<(), ApiError> {
    if state.queue_tx.len().saturating_add(jobs) <= state.config.queue_size {
        Ok(())
    } else {
        Err(ApiError::ServiceUnavailable(format!(
            "inference queue does not have enough capacity for {jobs} PDF pages"
        )))
    }
}

async fn uploaded_pending_submission(
    state: &AppState,
    request: MultipartInferenceRequest,
) -> Result<PendingSubmission, ApiError> {
    let task = task_spec_from_request(request.text_input);
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    let file = request
        .file
        .ok_or_else(|| ApiError::BadRequest("multipart field `image` is required".into()))?;
    if file.bytes.is_empty() {
        warn!("rejecting empty upload");
        return Err(ApiError::BadRequest("uploaded file is empty".into()));
    }
    if is_pdf_content(file.content_type.as_deref(), &file.bytes) {
        return uploaded_pdf_page_records(state, file, task, webhook_url)
            .await
            .map(PendingSubmission::Pdf);
    }

    validate_image_bytes(state, file.content_type.as_deref(), &file.bytes)?;

    let id = Uuid::new_v4();
    let now = OffsetDateTime::now_utc();
    let image_path = save_uploaded_image(&state.config, id, &file).await?;
    let image_sha256 = sha256_hex(&file.bytes);
    Ok(PendingSubmission::Image(Box::new(JobRecord {
        id,
        status: JobStatus::Queued,
        created_at: now,
        updated_at: now,
        image_path: image_path.clone(),
        filename: file.filename,
        content_type: file.content_type,
        image_sha256,
        image_bytes: file.bytes.len(),
        input_kind: "upload".to_string(),
        source_path: None,
        document_filename: None,
        document_page: None,
        document_pages: None,
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    })))
}

async fn uploaded_pdf_page_records(
    state: &AppState,
    pdf: UploadedFile,
    task: TaskSpec,
    webhook_url: Option<String>,
) -> Result<Vec<JobRecord>, ApiError> {
    let batch_id = Uuid::new_v4();
    let pdf_path = state.config.images_dir.join(format!("{batch_id}.pdf"));
    tokio::fs::write(&pdf_path, &pdf.bytes)
        .await
        .with_context(|| format!("failed to save uploaded PDF to {}", pdf_path.display()))?;

    let render_result = render_pdf_pages(
        &pdf_path,
        &state.config.images_dir,
        &format!("{batch_id}-page"),
        pdf_render_options(state),
    )
    .await;
    cleanup_uploaded_pdf(&pdf_path).await;
    let pages = render_result.map_err(pdf_render_api_error)?;

    page_records_from_rendered_pdf(
        state,
        pages,
        PdfRecordContext {
            input_kind: "upload_pdf_page",
            source_path: None,
            document_filename: pdf.filename,
            task,
            webhook_url,
        },
    )
    .await
}

async fn save_uploaded_image(
    config: &Config,
    id: Uuid,
    image: &UploadedFile,
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
) -> Result<Json<SubmissionResponse>, ApiError> {
    ensure_workers_ready(&state)?;
    debug!(
        "local path inference submission started image_path={}",
        request.image_path.display()
    );

    let submission = local_path_pending_submission(&state, request).await?;

    Ok(Json(enqueue_pending_submission(&state, submission).await?))
}

async fn local_path_pending_submission(
    state: &AppState,
    request: LocalPathRequest,
) -> Result<PendingSubmission, ApiError> {
    let image_path = validated_local_image_path(state, &request.image_path).await?;
    let bytes = read_local_image(&image_path).await?;
    let task = task_spec_from_request(request.text_input);
    let webhook_url = validate_webhook_url(&state.config, request.webhook_url)?;
    if is_pdf_content(None, &bytes) {
        return local_pdf_page_records(state, image_path, task, webhook_url)
            .await
            .map(PendingSubmission::Pdf);
    }

    let content_type = image::guess_format(&bytes)
        .ok()
        .and_then(image_format_content_type)
        .map(str::to_string);
    validate_image_bytes(state, content_type.as_deref(), &bytes)?;

    let id = Uuid::new_v4();
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
    Ok(PendingSubmission::Image(Box::new(JobRecord {
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
        document_filename: None,
        document_page: None,
        document_pages: None,
        text_input: task.text_input,
        webhook_url,
        result: None,
        error: None,
    })))
}

async fn local_pdf_page_records(
    state: &AppState,
    pdf_path: PathBuf,
    task: TaskSpec,
    webhook_url: Option<String>,
) -> Result<Vec<JobRecord>, ApiError> {
    let batch_id = Uuid::new_v4();
    let pages = render_pdf_pages(
        &pdf_path,
        &state.config.images_dir,
        &format!("{batch_id}-page"),
        pdf_render_options(state),
    )
    .await
    .map_err(pdf_render_api_error)?;
    let document_filename = pdf_path
        .file_name()
        .map(|filename| filename.to_string_lossy().into_owned());

    page_records_from_rendered_pdf(
        state,
        pages,
        PdfRecordContext {
            input_kind: "local_pdf_page",
            source_path: Some(pdf_path),
            document_filename,
            task,
            webhook_url,
        },
    )
    .await
}

struct PdfRecordContext {
    input_kind: &'static str,
    source_path: Option<PathBuf>,
    document_filename: Option<String>,
    task: TaskSpec,
    webhook_url: Option<String>,
}

async fn page_records_from_rendered_pdf(
    state: &AppState,
    pages: Vec<RenderedPdfPage>,
    context: PdfRecordContext,
) -> Result<Vec<JobRecord>, ApiError> {
    let page_count = pages.len();
    let page_paths = pages
        .iter()
        .map(|page| page.image_path.clone())
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(page_count);
    for page in pages {
        let bytes = match read_rendered_page(&page.image_path).await {
            Ok(bytes) => bytes,
            Err(err) => {
                cleanup_paths(&page_paths).await;
                return Err(err);
            }
        };
        if let Err(err) = validate_image_bytes(state, Some("image/png"), &bytes) {
            cleanup_paths(&page_paths).await;
            return Err(err);
        }

        let id = Uuid::new_v4();
        let now = OffsetDateTime::now_utc();
        records.push(JobRecord {
            id,
            status: JobStatus::Queued,
            created_at: now,
            updated_at: now,
            image_path: page.image_path,
            filename: context
                .document_filename
                .as_ref()
                .map(|filename| format!("{filename}#page={}", page.page_number)),
            content_type: Some("image/png".to_string()),
            image_sha256: sha256_hex(&bytes),
            image_bytes: bytes.len(),
            input_kind: context.input_kind.to_string(),
            source_path: context.source_path.clone(),
            document_filename: context.document_filename.clone(),
            document_page: Some(page.page_number),
            document_pages: Some(page_count),
            text_input: context.task.text_input.clone(),
            webhook_url: context.webhook_url.clone(),
            result: None,
            error: None,
        });
    }
    Ok(records)
}

async fn read_rendered_page(page_path: &PathBuf) -> Result<Vec<u8>, ApiError> {
    tokio::fs::read(page_path).await.map_err(|err| {
        ApiError::BadRequest(format!(
            "failed to read rendered PDF page {}: {err}",
            page_path.display()
        ))
    })
}

async fn cleanup_uploaded_pdf(pdf_path: &PathBuf) {
    if let Err(err) = tokio::fs::remove_file(pdf_path).await {
        warn!(
            "failed to remove temporary uploaded PDF path={} error={}",
            pdf_path.display(),
            err
        );
    }
}

async fn cleanup_rendered_pages(records: &[JobRecord]) {
    let paths = records
        .iter()
        .map(|record| record.image_path.clone())
        .collect::<Vec<_>>();
    cleanup_paths(&paths).await;
}

async fn cleanup_paths(paths: &[PathBuf]) {
    for path in paths {
        let _ = tokio::fs::remove_file(path).await;
    }
}

fn pdf_render_options(state: &AppState) -> PdfRenderOptions {
    PdfRenderOptions {
        max_pages: state.config.max_pdf_pages,
        dpi: state.config.pdf_render_dpi,
    }
}

fn pdf_render_api_error(err: PdfRenderError) -> ApiError {
    match err {
        PdfRenderError::RendererUnavailable | PdfRenderError::Io(_) => ApiError::Internal(anyhow!(
            "PDF processing requires poppler-utils (`pdfinfo` and `pdftoppm`): {err}"
        )),
        PdfRenderError::Inspect(_)
        | PdfRenderError::TooManyPages { .. }
        | PdfRenderError::Render(_) => ApiError::BadRequest(err.to_string()),
    }
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
