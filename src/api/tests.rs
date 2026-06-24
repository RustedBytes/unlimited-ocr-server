use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc};

use async_channel::bounded;
use axum::body::Body;
use axum::http::{HeaderName, HeaderValue, header};
use axum::middleware;
use serde_json::json;
use tokio::sync::RwLock;
use tower::ServiceExt;
use tower_http::timeout::TimeoutLayer;

use super::submit::{
    ensure_local_path_allowed, task_spec_from_request, validate_image_bytes, validate_webhook_url,
};
use super::system::parse_nvidia_smi_memory;
use super::*;
use crate::{
    config::{Config, ModelVariant},
    state::{AppMetrics, RateLimiter, WorkerPoolState},
};

#[test]
fn request_task_defaults_to_default_prompt() {
    let task = task_spec_from_request(None);

    assert_eq!(task.text_input, None);
}

#[test]
fn request_task_trims_text_input() {
    let task = task_spec_from_request(Some("  text to trim  ".to_string()));

    assert_eq!(task.text_input.as_deref(), Some("text to trim"));
}

#[test]
fn request_task_ignores_empty_text_input() {
    let task = task_spec_from_request(Some("   ".to_string()));

    assert_eq!(task.text_input, None);
}

#[test]
fn local_path_request_rejects_removed_task_fields() {
    let err = serde_json::from_value::<submit::LocalPathRequest>(json!({
        "image_path": "/tmp/image.png",
        "task": "OCR"
    }))
    .unwrap_err();

    assert!(err.to_string().contains("unknown field `task`"));
}

#[test]
fn index_template_contains_unlimited_ocr_copy_and_error() {
    let html = render_index(None, Some("bad request".to_string()))
        .unwrap()
        .0;

    assert!(html.contains("Unlimited-OCR Inference"));
    assert!(html.contains("Prompt (optional)"));
    assert!(html.contains("bad request"));
}

#[test]
fn api_errors_have_stable_codes() {
    assert_eq!(ApiError::NotFound.code(), "not_found");
    assert_eq!(
        ApiError::ServiceUnavailable("not ready".to_string()).code(),
        "service_unavailable"
    );
}

#[test]
fn openapi_document_describes_core_paths() {
    let document = openapi_document();

    assert_eq!(document["openapi"], "3.1.0");
    assert!(document["paths"]["/v1/infer"].is_object());
    assert!(document["paths"]["/metrics"].is_object());
    assert!(document["components"]["schemas"]["ErrorResponse"].is_object());
    assert!(document["components"]["schemas"]["LocalPathRequest"]["properties"]["task"].is_null());
    assert!(
        document["components"]["schemas"]["UploadInferenceRequest"]["properties"]["task_prompt"]
            .is_null()
    );
}

#[test]
fn validates_supported_image_bytes() {
    let state = test_state(false, Vec::new());

    validate_image_bytes(&state, Some("image/png"), ONE_BY_ONE_PNG).unwrap();
}

#[test]
fn rejects_non_image_content_type() {
    let state = test_state(false, Vec::new());
    let err = validate_image_bytes(&state, Some("text/plain"), ONE_BY_ONE_PNG).unwrap_err();

    assert!(matches!(err, ApiError::BadRequest(_)));
    assert!(err.to_string().contains("unsupported content type"));
}

#[test]
fn validates_optional_webhook_url() {
    let state = test_state(false, Vec::new());
    let url =
        validate_webhook_url(&state.config, Some(" http://example.com/hook ".to_string())).unwrap();

    assert_eq!(url.as_deref(), Some("http://example.com/hook"));
    assert_eq!(
        validate_webhook_url(&state.config, Some("  ".to_string())).unwrap(),
        None
    );
}

#[test]
fn rejects_unsupported_webhook_url_scheme() {
    let state = test_state(false, Vec::new());
    let err =
        validate_webhook_url(&state.config, Some("file:///tmp/hook".to_string())).unwrap_err();

    assert!(matches!(err, ApiError::BadRequest(_)));
    assert!(err.to_string().contains("expected http or https"));
}

#[test]
fn rejects_webhook_url_credentials() {
    let state = test_state(false, Vec::new());
    let err = validate_webhook_url(
        &state.config,
        Some("https://user:secret@example.com/hook".to_string()),
    )
    .unwrap_err();

    assert!(matches!(err, ApiError::BadRequest(_)));
    assert!(err.to_string().contains("must not include credentials"));
}

#[test]
fn rejects_private_webhook_urls_by_default() {
    let state = test_state(false, Vec::new());
    let err =
        validate_webhook_url(&state.config, Some("http://127.0.0.1/hook".to_string())).unwrap_err();

    assert!(matches!(err, ApiError::BadRequest(_)));
    assert!(err.to_string().contains("private or local address"));
}

#[test]
fn allows_private_webhook_urls_when_configured() {
    let mut state = test_state(false, Vec::new());
    Arc::get_mut(&mut state.config)
        .unwrap()
        .allow_private_webhook_urls = true;

    let url =
        validate_webhook_url(&state.config, Some("http://127.0.0.1/hook".to_string())).unwrap();

    assert_eq!(url.as_deref(), Some("http://127.0.0.1/hook"));
}

#[tokio::test]
async fn metrics_response_reports_counters() {
    let state = test_state(false, Vec::new());
    state.metrics.record_job_started();
    state.metrics.record_job_succeeded(42);

    let Json(response) = metrics(State(state)).await;

    assert_eq!(response.jobs_started, 1);
    assert_eq!(response.jobs_succeeded, 1);
    assert_eq!(response.total_inference_ms, 42);
    assert_eq!(response.average_inference_ms, Some(42.0));
}

#[tokio::test]
async fn api_key_auth_is_disabled_without_configured_keys() {
    let state = test_state(false, Vec::new());
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn api_key_auth_rejects_missing_key_when_enabled() {
    let mut state = test_state(false, Vec::new());
    Arc::get_mut(&mut state.config).unwrap().api_keys = vec!["secret".to_string()];
    let metrics = Arc::clone(&state.metrics);
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.http_requests_total, 1);
    assert_eq!(snapshot.http_requests_failed, 1);
}

#[tokio::test]
async fn api_key_auth_accepts_x_api_key() {
    let mut state = test_state(false, Vec::new());
    Arc::get_mut(&mut state.config).unwrap().api_keys = vec!["secret".to_string()];
    let app = router(state);

    let response = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .header("x-api-key", "secret")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn rate_limit_rejects_after_configured_limit() {
    let mut state = test_state(false, Vec::new());
    Arc::get_mut(&mut state.config)
        .unwrap()
        .rate_limit_requests_per_minute = 1;
    state.rate_limiter = Arc::new(RateLimiter::new(1));
    let app = router(state);

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let second = app
        .oneshot(
            Request::builder()
                .uri("/openapi.json")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(first.status(), StatusCode::OK);
    assert_eq!(second.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(second.headers().contains_key(header::RETRY_AFTER));
}

#[test]
fn parses_nvidia_smi_memory_totals() {
    let memory = parse_nvidia_smi_memory("100, 1000\n25, 500\n").unwrap();

    assert_eq!(memory.used_bytes, 125 * 1024 * 1024);
    assert_eq!(memory.total_bytes, 1500 * 1024 * 1024);
}

#[tokio::test]
async fn request_timeout_layer_returns_408() {
    let app = Router::new()
        .route(
            "/slow",
            get(|| async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                "ok"
            }),
        )
        .layer(TimeoutLayer::with_status_code(
            StatusCode::REQUEST_TIMEOUT,
            Duration::from_millis(1),
        ));

    let response = app
        .oneshot(Request::builder().uri("/slow").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
}

#[tokio::test]
async fn security_headers_are_added_to_responses() {
    let app = Router::new()
        .route("/ok", get(|| async { "ok" }))
        .layer(middleware::from_fn(add_security_headers));

    let response = app
        .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
        .await
        .unwrap();

    assert_eq!(
        response.headers().get(header::X_CONTENT_TYPE_OPTIONS),
        Some(&HeaderValue::from_static("nosniff"))
    );
    assert_eq!(
        response
            .headers()
            .get(HeaderName::from_static("x-frame-options")),
        Some(&HeaderValue::from_static("DENY"))
    );
    assert!(
        response
            .headers()
            .contains_key(HeaderName::from_static("content-security-policy"))
    );
}

#[tokio::test]
async fn local_path_endpoint_is_disabled_by_default() {
    let state = test_state(false, Vec::new());
    let path = std::env::temp_dir().join(format!("unlimited-ocr-api-test-{}", Uuid::new_v4()));
    tokio::fs::write(&path, b"image").await.unwrap();
    let path = tokio::fs::canonicalize(&path).await.unwrap();

    let err = ensure_local_path_allowed(&state, &path).await.unwrap_err();

    assert!(matches!(err, ApiError::Forbidden(_)));
    assert!(err.to_string().contains("local path inference is disabled"));

    tokio::fs::remove_file(path).await.unwrap();
}

#[tokio::test]
async fn local_path_endpoint_allows_files_under_configured_roots() {
    let root = std::env::temp_dir().join(format!("unlimited-ocr-api-root-{}", Uuid::new_v4()));
    tokio::fs::create_dir_all(&root).await.unwrap();
    let image_path = root.join("image.bin");
    tokio::fs::write(&image_path, b"image").await.unwrap();
    let image_path = tokio::fs::canonicalize(&image_path).await.unwrap();
    let state = test_state(true, vec![root.clone()]);

    ensure_local_path_allowed(&state, &image_path)
        .await
        .unwrap();

    tokio::fs::remove_dir_all(root).await.unwrap();
}

fn test_state(allow_local_paths: bool, local_path_roots: Vec<PathBuf>) -> AppState {
    let (queue_tx, _queue_rx) = bounded(1);
    AppState {
        config: Arc::new(Config {
            addr: SocketAddr::from(([127, 0, 0, 1], 3000)),
            model_path: PathBuf::from("Unlimited-OCR/onnx/unlimited_ocr.onnx"),
            decode_model_path: None,
            model_variant: ModelVariant::UnlimitedOcr,
            model_image_size: 1024,
            data_dir: PathBuf::from("data"),
            images_dir: PathBuf::from("data/images"),
            metadata_dir: PathBuf::from("data/metadata"),
            submissions_jsonl: PathBuf::from("data/metadata/submissions.jsonl"),
            results_jsonl: PathBuf::from("data/metadata/results.jsonl"),
            allow_local_paths,
            local_path_roots,
            cors_allowed_origins: Vec::new(),
            api_keys: Vec::new(),
            rate_limit_requests_per_minute: 0,
            job_retention_limit: 1000,
            metadata_retention_limit: 10_000,
            max_image_width: 8192,
            max_image_height: 8192,
            workers: 1,
            queue_size: 1,
            body_limit_bytes: 1024,
            request_timeout_seconds: 60,
            rust_log: "info".to_string(),
            max_new_tokens: 1,
            job_timeout_seconds: 300,
            webhook_timeout_seconds: 10,
            webhook_connect_timeout_seconds: 5,
            webhook_max_attempts: 1,
            webhook_initial_backoff_ms: 500,
            webhook_signing_secret: None,
            webhooks_dead_letter_jsonl: PathBuf::from("data/metadata/webhooks_dead_letter.jsonl"),
            allow_private_webhook_urls: false,
            execution_providers: vec!["cpu".to_string()],
        }),
        queue_tx,
        jobs: Arc::new(RwLock::new(HashMap::new())),
        workers: Arc::new(WorkerPoolState::new(1)),
        metrics: Arc::new(AppMetrics::default()),
        rate_limiter: Arc::new(RateLimiter::new(0)),
    }
}

const ONE_BY_ONE_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 13, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13, 10,
    45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];
