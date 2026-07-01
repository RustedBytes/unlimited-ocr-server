use std::{collections::HashMap, net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};

use anyhow::anyhow;
use async_channel::bounded;
use time::OffsetDateTime;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::*;
use crate::{
    config::{Config, ModelVariant},
    state::{AppMetrics, AppState, RateLimiter, WorkerPoolState},
    types::{JobRecord, JobStatus, TaskSpec},
    util::append_jsonl,
};

fn test_config() -> Config {
    Config {
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
        allow_local_paths: false,
        local_path_roots: Vec::new(),
        cors_allowed_origins: Vec::new(),
        api_keys: Vec::new(),
        rate_limit_requests_per_minute: 0,
        job_retention_limit: 1000,
        metadata_retention_limit: 10_000,
        max_image_width: 8192,
        max_image_height: 8192,
        max_pdf_pages: 32,
        pdf_render_dpi: 200,
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
    }
}

fn job_record(input_kind: &str, image_path: PathBuf) -> JobRecord {
    let now = OffsetDateTime::now_utc();

    JobRecord {
        id: Uuid::nil(),
        status: JobStatus::Succeeded,
        created_at: now,
        updated_at: now,
        image_path,
        filename: None,
        content_type: None,
        image_sha256: String::new(),
        image_bytes: 0,
        input_kind: input_kind.to_string(),
        source_path: None,
        document_filename: None,
        document_page: None,
        document_pages: None,
        text_input: None,
        webhook_url: None,
        result: None,
        error: None,
    }
}

fn temp_config(name: &str) -> Config {
    let mut config = test_config();
    let dir = std::env::temp_dir().join(format!("unlimited-ocr-server-{name}-{}", Uuid::new_v4()));
    std::fs::create_dir_all(dir.join("metadata")).unwrap();
    config.data_dir = dir.clone();
    config.images_dir = dir.join("images");
    config.metadata_dir = dir.join("metadata");
    config.submissions_jsonl = config.metadata_dir.join("submissions.jsonl");
    config.results_jsonl = config.metadata_dir.join("results.jsonl");
    config.webhooks_dead_letter_jsonl = config.metadata_dir.join("webhooks_dead_letter.jsonl");
    config
}

#[test]
fn deletes_only_uploaded_images_under_images_dir() {
    let config = test_config();

    assert!(should_delete_image(
        &config,
        &job_record("upload", PathBuf::from("data/images/job.png"))
    ));
    assert!(!should_delete_image(
        &config,
        &job_record("local_path", PathBuf::from("data/images/job.png"))
    ));
    assert!(should_delete_image(
        &config,
        &job_record("upload_pdf_page", PathBuf::from("data/images/page.png"))
    ));
    assert!(should_delete_image(
        &config,
        &job_record("local_pdf_page", PathBuf::from("data/images/page.png"))
    ));
    assert!(!should_delete_image(
        &config,
        &job_record("upload", PathBuf::from("/tmp/source.png"))
    ));
    assert!(!should_delete_image(
        &config,
        &job_record("upload", PathBuf::from("data/images"))
    ));
}

#[tokio::test]
async fn load_jobs_recovers_non_terminal_jobs_as_failed() {
    let config = temp_config("recover");
    let mut record = job_record("upload", config.images_dir.join("job.png"));
    record.status = JobStatus::Running;

    append_jsonl(&config.submissions_jsonl, &record)
        .await
        .unwrap();

    let jobs = load_jobs(&config).await.unwrap();
    let recovered = jobs.get(&record.id).unwrap();

    assert_eq!(recovered.status, JobStatus::Failed);
    assert_eq!(
        recovered.error.as_deref(),
        Some("server restarted before job reached a terminal state")
    );
    assert!(
        std::fs::read_to_string(&config.results_jsonl)
            .unwrap()
            .contains("server restarted before job reached a terminal state")
    );

    std::fs::remove_dir_all(&config.data_dir).unwrap();
}

#[tokio::test]
async fn load_jobs_applies_retention_and_compacts_metadata() {
    let mut config = temp_config("retention");
    config.job_retention_limit = 2;
    config.metadata_retention_limit = 2;

    for minutes in 0..3 {
        let mut record = job_record("upload", config.images_dir.join(format!("{minutes}.png")));
        record.id = Uuid::new_v4();
        record.status = JobStatus::Succeeded;
        record.updated_at += time::Duration::minutes(minutes);
        append_jsonl(&config.results_jsonl, &record).await.unwrap();
    }

    let jobs = load_jobs(&config).await.unwrap();
    let compacted = std::fs::read_to_string(&config.results_jsonl).unwrap();

    assert_eq!(jobs.len(), 2);
    assert_eq!(compacted.lines().count(), 2);

    std::fs::remove_dir_all(&config.data_dir).unwrap();
}

#[tokio::test]
async fn enqueue_record_rejects_full_queue_without_persisting() {
    let config = Arc::new(temp_config("queue-full"));
    let (queue_tx, queue_rx) = bounded(1);
    queue_tx
        .try_send(JobRequest {
            id: Uuid::new_v4(),
            image_path: PathBuf::from("busy.png"),
            task: TaskSpec::default(),
        })
        .unwrap();
    let state = AppState {
        config: Arc::clone(&config),
        queue_tx,
        jobs: Arc::new(RwLock::new(HashMap::new())),
        workers: Arc::new(WorkerPoolState::new(1)),
        metrics: Arc::new(AppMetrics::default()),
        rate_limiter: Arc::new(RateLimiter::new(0)),
    };
    let record = job_record("upload", config.images_dir.join("job.png"));
    let err = enqueue_record(&state, record.clone()).await.unwrap_err();

    assert!(matches!(err, EnqueueError::QueueFull));
    assert!(state.jobs.read().await.get(&record.id).is_none());
    assert!(!config.submissions_jsonl.exists());

    std::fs::remove_dir_all(&config.data_dir).unwrap();
    drop(queue_rx);
}

#[tokio::test]
async fn post_webhook_sends_final_job_record() {
    let (url, server) = spawn_webhook_receiver().await;
    let mut record = job_record("upload", PathBuf::from("data/images/job.png"));
    record.id = Uuid::new_v4();
    record.webhook_url = Some(url.clone());

    WebhookClient::new(
        10,
        5,
        1,
        Duration::from_millis(1),
        Some("secret".to_string()),
        std::env::temp_dir().join("unused-webhook-dead-letter.jsonl"),
    )
    .unwrap()
    .send(&WebhookEvent::from_job(record.clone()), 1)
    .await
    .unwrap();

    let request = server.await.unwrap();
    let request = String::from_utf8(request).unwrap();
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body = serde_json::from_str::<serde_json::Value>(body).unwrap();

    assert!(request.starts_with("POST /hook HTTP/1.1"));
    assert_eq!(body["event_type"], "job.completed");
    assert_eq!(body["job"]["id"], record.id.to_string());
    assert_eq!(body["job"]["status"], "succeeded");
    assert_eq!(body["job"]["webhook_url"], url);
    assert!(request.contains("x-server-event-id:"));
    assert!(request.contains("x-server-delivery-attempt: 1"));
    assert!(request.contains("x-server-signature: sha256="));
}

#[test]
fn redacted_webhook_url_removes_sensitive_parts() {
    let redacted = redacted_webhook_url("https://user:secret@example.com/hook?token=secret#frag");

    assert_eq!(redacted, "https://example.com/hook");
}

#[tokio::test]
async fn finish_job_delivers_webhook_with_terminal_record() {
    let config = temp_config("finish-webhook");
    let (url, server) = spawn_webhook_receiver().await;
    let mut record = job_record("upload", config.images_dir.join("job.png"));
    record.id = Uuid::new_v4();
    record.status = JobStatus::Running;
    record.webhook_url = Some(url);
    let id = record.id;
    let jobs = RwLock::new(HashMap::from([(id, record)]));

    let metrics = AppMetrics::default();
    let webhooks = WebhookClient::new(
        10,
        5,
        1,
        Duration::from_millis(1),
        None,
        config.webhooks_dead_letter_jsonl.clone(),
    )
    .unwrap();
    finish_job(
        &config,
        &jobs,
        &metrics,
        &webhooks,
        id,
        Err(anyhow!("inference failed")),
    )
    .await;

    let request = server.await.unwrap();
    let request = String::from_utf8(request).unwrap();
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body = serde_json::from_str::<serde_json::Value>(body).unwrap();

    assert_eq!(body["job"]["id"], id.to_string());
    assert_eq!(body["job"]["status"], "failed");
    assert_eq!(body["job"]["error"], "inference failed");

    std::fs::remove_dir_all(&config.data_dir).unwrap();
}

#[tokio::test]
async fn webhook_delivery_retries_with_same_event_id() {
    let config = temp_config("webhook-retry");
    let (url, server) =
        spawn_webhook_status_sequence(["500 Internal Server Error", "204 No Content"]).await;
    let mut record = job_record("upload", config.images_dir.join("job.png"));
    record.id = Uuid::new_v4();
    record.webhook_url = Some(url);
    let webhooks = WebhookClient::new(
        10,
        5,
        2,
        Duration::from_millis(1),
        None,
        config.webhooks_dead_letter_jsonl.clone(),
    )
    .unwrap();

    let result = webhooks.deliver(&record).await;
    let requests = server.await.unwrap();

    assert!(matches!(
        result,
        WebhookDeliveryResult::Delivered { attempts: 2, .. }
    ));
    assert_eq!(requests.len(), 2);
    assert_eq!(
        header_value(&requests[0], "x-server-event-id"),
        header_value(&requests[1], "x-server-event-id")
    );
    assert!(!config.webhooks_dead_letter_jsonl.exists());

    std::fs::remove_dir_all(&config.data_dir).unwrap();
}

#[tokio::test]
async fn webhook_delivery_writes_dead_letter_after_final_failure() {
    let config = temp_config("webhook-dead-letter");
    let (url, server) =
        spawn_webhook_status_sequence(["500 Internal Server Error", "500 Internal Server Error"])
            .await;
    let mut record = job_record("upload", config.images_dir.join("job.png"));
    record.id = Uuid::new_v4();
    record.webhook_url = Some(url);
    let webhooks = WebhookClient::new(
        10,
        5,
        2,
        Duration::from_millis(1),
        None,
        config.webhooks_dead_letter_jsonl.clone(),
    )
    .unwrap();

    let result = webhooks.deliver(&record).await;
    let requests = server.await.unwrap();
    let dead_letter = std::fs::read_to_string(&config.webhooks_dead_letter_jsonl).unwrap();

    assert!(matches!(
        result,
        WebhookDeliveryResult::Failed { attempts: 2, .. }
    ));
    assert_eq!(requests.len(), 2);
    assert!(dead_letter.contains("\"attempts\":2"));
    assert!(dead_letter.contains(&record.id.to_string()));
    assert!(dead_letter.contains("webhook endpoint returned HTTP 500"));

    std::fs::remove_dir_all(&config.data_dir).unwrap();
}

async fn spawn_webhook_receiver() -> (String, tokio::task::JoinHandle<Vec<u8>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/hook", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];

        loop {
            let n = stream.read(&mut buffer).await.unwrap();
            assert!(n > 0, "client closed before sending full request");
            request.extend_from_slice(&buffer[..n]);

            if request_is_complete(&request) {
                break;
            }
        }

        stream
            .write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n")
            .await
            .unwrap();
        request
    });

    (url, server)
}

async fn spawn_webhook_status_sequence<const N: usize>(
    statuses: [&'static str; N],
) -> (String, tokio::task::JoinHandle<Vec<Vec<u8>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/hook", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let mut requests = Vec::new();
        for status in statuses {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];

            loop {
                let n = stream.read(&mut buffer).await.unwrap();
                assert!(n > 0, "client closed before sending full request");
                request.extend_from_slice(&buffer[..n]);

                if request_is_complete(&request) {
                    break;
                }
            }

            let response = format!("HTTP/1.1 {status}\r\nContent-Length: 0\r\n\r\n");
            stream.write_all(response.as_bytes()).await.unwrap();
            requests.push(request);
        }
        requests
    });

    (url, server)
}

fn header_value(request: &[u8], name: &str) -> Option<String> {
    let headers = String::from_utf8_lossy(request);
    headers
        .lines()
        .find_map(|line| line.split_once(':'))
        .filter(|(header_name, _)| header_name.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.trim().to_string())
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|position| position + 4)
    else {
        return false;
    };

    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        .unwrap_or(0);

    request.len() >= header_end + content_length
}
