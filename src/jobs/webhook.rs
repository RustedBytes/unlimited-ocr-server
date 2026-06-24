use std::{path::PathBuf, time::Duration};

use anyhow::{Context, anyhow};
use log::{info, warn};
use serde::Serialize;
use uuid::Uuid;

use crate::{
    config::Config,
    state::AppMetrics,
    types::JobRecord,
    util::{append_jsonl, hmac_sha256_hex},
};

#[derive(Clone)]
pub struct WebhookClient {
    http: reqwest::Client,
    max_attempts: usize,
    initial_backoff: Duration,
    signing_secret: Option<String>,
    dead_letter_jsonl: PathBuf,
}

impl WebhookClient {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        Self::new(
            config.webhook_timeout_seconds,
            config.webhook_connect_timeout_seconds,
            config.webhook_max_attempts,
            Duration::from_millis(config.webhook_initial_backoff_ms),
            config.webhook_signing_secret.clone(),
            config.webhooks_dead_letter_jsonl.clone(),
        )
    }

    pub(super) fn new(
        timeout_seconds: u64,
        connect_timeout_seconds: u64,
        max_attempts: usize,
        initial_backoff: Duration,
        signing_secret: Option<String>,
        dead_letter_jsonl: PathBuf,
    ) -> anyhow::Result<Self> {
        let mut builder = reqwest::Client::builder().redirect(reqwest::redirect::Policy::none());
        if timeout_seconds > 0 {
            builder = builder.timeout(Duration::from_secs(timeout_seconds));
        }
        if connect_timeout_seconds > 0 {
            builder = builder.connect_timeout(Duration::from_secs(connect_timeout_seconds));
        }

        let http = builder
            .build()
            .context("failed to build webhook HTTP client")?;

        Ok(Self {
            http,
            max_attempts: max_attempts.max(1),
            initial_backoff,
            signing_secret,
            dead_letter_jsonl,
        })
    }

    pub(super) async fn send(&self, event: &WebhookEvent, attempt: usize) -> anyhow::Result<()> {
        let record = &event.job;
        let Some(webhook_url) = record.webhook_url.as_deref() else {
            return Ok(());
        };
        let redacted_url = redacted_webhook_url(webhook_url);
        let body = serde_json::to_vec(event).context("failed to serialize webhook event")?;

        let mut request = self
            .http
            .post(webhook_url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header("x-server-event-id", event.event_id.to_string())
            .header("x-server-event-type", event.event_type)
            .header("x-server-delivery-attempt", attempt.to_string())
            .body(body.clone());
        if let Some(secret) = &self.signing_secret {
            request = request.header(
                "x-server-signature",
                format!("sha256={}", hmac_sha256_hex(secret, &body)),
            );
        }

        let response = request
            .send()
            .await
            .with_context(|| format!("failed to send webhook request to {redacted_url}"))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "webhook endpoint returned HTTP {}",
                response.status()
            ));
        }

        Ok(())
    }

    pub(super) async fn deliver(&self, record: &JobRecord) -> WebhookDeliveryResult {
        if record.webhook_url.is_none() {
            return WebhookDeliveryResult::Skipped;
        }

        let event = WebhookEvent::from_job(record.clone());
        let mut last_error = None;

        for attempt in 1..=self.max_attempts {
            match self.send(&event, attempt).await {
                Ok(()) => {
                    return WebhookDeliveryResult::Delivered {
                        event_id: event.event_id,
                        attempts: attempt,
                    };
                }
                Err(err) => {
                    last_error = Some(err.to_string());
                    if attempt < self.max_attempts && !self.initial_backoff.is_zero() {
                        tokio::time::sleep(backoff_delay(self.initial_backoff, attempt)).await;
                    }
                }
            }
        }

        let error = last_error.unwrap_or_else(|| "webhook delivery failed".to_string());
        let dead_letter = WebhookDeadLetter {
            event,
            attempts: self.max_attempts,
            failed_at: time::OffsetDateTime::now_utc(),
            error: error.clone(),
        };
        match append_jsonl(&self.dead_letter_jsonl, &dead_letter).await {
            Ok(()) => WebhookDeliveryResult::Failed {
                event_id: dead_letter.event.event_id,
                attempts: dead_letter.attempts,
                error,
            },
            Err(err) => WebhookDeliveryResult::DeadLetterFailed {
                event_id: dead_letter.event.event_id,
                attempts: dead_letter.attempts,
                error,
                dead_letter_error: err.to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WebhookEvent {
    pub event_id: Uuid,
    pub event_type: &'static str,
    #[serde(with = "time::serde::rfc3339")]
    pub created_at: time::OffsetDateTime,
    pub job: JobRecord,
}

impl WebhookEvent {
    pub(super) fn from_job(job: JobRecord) -> Self {
        Self {
            event_id: Uuid::new_v4(),
            event_type: "job.completed",
            created_at: time::OffsetDateTime::now_utc(),
            job,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct WebhookDeadLetter {
    pub event: WebhookEvent,
    pub attempts: usize,
    #[serde(with = "time::serde::rfc3339")]
    pub failed_at: time::OffsetDateTime,
    pub error: String,
}

pub(super) enum WebhookDeliveryResult {
    Skipped,
    Delivered {
        event_id: Uuid,
        attempts: usize,
    },
    Failed {
        event_id: Uuid,
        attempts: usize,
        error: String,
    },
    DeadLetterFailed {
        event_id: Uuid,
        attempts: usize,
        error: String,
        dead_letter_error: String,
    },
}

pub(super) async fn send_webhook(
    metrics: &AppMetrics,
    webhooks: &WebhookClient,
    record: &JobRecord,
) {
    let Some(webhook_url) = record.webhook_url.as_deref() else {
        return;
    };
    let redacted_url = redacted_webhook_url(webhook_url);

    match webhooks.deliver(record).await {
        WebhookDeliveryResult::Skipped => {}
        WebhookDeliveryResult::Delivered { event_id, attempts } => info!(
            "webhook delivered job_id={} event_id={} url={} attempts={}",
            record.id, event_id, redacted_url, attempts
        ),
        WebhookDeliveryResult::Failed {
            event_id,
            attempts,
            error,
        } => {
            metrics.record_webhook_failure();
            warn!(
                "webhook delivery failed job_id={} event_id={} url={} attempts={} error={}",
                record.id, event_id, redacted_url, attempts, error
            );
        }
        WebhookDeliveryResult::DeadLetterFailed {
            event_id,
            attempts,
            error,
            dead_letter_error,
        } => {
            metrics.record_webhook_failure();
            warn!(
                "webhook delivery failed and dead-letter append failed job_id={} event_id={} url={} attempts={} error={} dead_letter_error={}",
                record.id, event_id, redacted_url, attempts, error, dead_letter_error
            );
        }
    }
}

pub(super) fn redacted_webhook_url(webhook_url: &str) -> String {
    let Ok(mut url) = reqwest::Url::parse(webhook_url) else {
        return "<invalid webhook url>".to_string();
    };
    url.set_query(None);
    url.set_fragment(None);
    if url.set_username("").is_err() {
        warn!("failed to redact webhook URL username");
        return "<redacted webhook url>".to_string();
    }
    if url.set_password(None).is_err() {
        warn!("failed to redact webhook URL password");
        return "<redacted webhook url>".to_string();
    }
    url.to_string()
}

fn backoff_delay(initial_backoff: Duration, attempt: usize) -> Duration {
    let multiplier = 1_u32.checked_shl((attempt - 1).min(16) as u32).unwrap_or(1);
    initial_backoff.saturating_mul(multiplier)
}
