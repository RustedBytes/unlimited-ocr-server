use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use async_channel::Sender;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{config::Config, jobs::JobRequest, types::JobRecord};

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub queue_tx: Sender<JobRequest>,
    pub jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    pub workers: Arc<WorkerPoolState>,
    pub metrics: Arc<AppMetrics>,
    pub rate_limiter: Arc<RateLimiter>,
}

#[derive(Debug)]
pub struct WorkerPoolState {
    expected: usize,
    ready: AtomicUsize,
    failed: AtomicUsize,
}

#[derive(Debug, Default)]
pub struct AppMetrics {
    jobs_started: AtomicUsize,
    jobs_succeeded: AtomicUsize,
    jobs_failed: AtomicUsize,
    jobs_timed_out: AtomicUsize,
    worker_restarts: AtomicUsize,
    total_inference_ms: AtomicUsize,
    model_load_ms: AtomicUsize,
    http_requests_total: AtomicUsize,
    http_requests_failed: AtomicUsize,
    total_request_ms: AtomicUsize,
    webhook_failures: AtomicUsize,
    cleanup_failures: AtomicUsize,
}

#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub jobs_started: usize,
    pub jobs_succeeded: usize,
    pub jobs_failed: usize,
    pub jobs_timed_out: usize,
    pub worker_restarts: usize,
    pub total_inference_ms: usize,
    pub model_load_ms: Option<usize>,
    pub http_requests_total: usize,
    pub http_requests_failed: usize,
    pub total_request_ms: usize,
    pub webhook_failures: usize,
    pub cleanup_failures: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct WorkerPoolSnapshot {
    pub expected: usize,
    pub ready: usize,
    pub failed: usize,
}

#[derive(Debug)]
pub struct RateLimiter {
    limit_per_minute: u64,
    state: Mutex<RateLimitWindow>,
}

#[derive(Debug)]
struct RateLimitWindow {
    started_at: Instant,
    count: u64,
}

impl WorkerPoolState {
    pub fn new(expected: usize) -> Self {
        Self {
            expected,
            ready: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        }
    }

    pub fn mark_ready(&self) {
        self.ready.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_stopped(&self) {
        let _ = self
            .ready
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |ready| {
                ready.checked_sub(1)
            });
    }

    pub fn snapshot(&self) -> WorkerPoolSnapshot {
        WorkerPoolSnapshot {
            expected: self.expected,
            ready: self.ready.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
        }
    }

    pub fn is_ready(&self) -> bool {
        self.snapshot().ready > 0
    }
}

impl RateLimiter {
    pub fn new(limit_per_minute: u64) -> Self {
        Self {
            limit_per_minute,
            state: Mutex::new(RateLimitWindow {
                started_at: Instant::now(),
                count: 0,
            }),
        }
    }

    pub fn check(&self) -> RateLimitDecision {
        if self.limit_per_minute == 0 {
            return RateLimitDecision::Allowed;
        }

        let mut state = self
            .state
            .lock()
            .expect("rate limiter mutex should not be poisoned");
        let now = Instant::now();
        let elapsed = now.duration_since(state.started_at);
        if elapsed >= Duration::from_secs(60) {
            state.started_at = now;
            state.count = 0;
        }

        if state.count >= self.limit_per_minute {
            let retry_after = Duration::from_secs(60)
                .checked_sub(now.duration_since(state.started_at))
                .unwrap_or_default()
                .as_secs()
                .max(1);
            return RateLimitDecision::Limited { retry_after };
        }

        state.count += 1;
        RateLimitDecision::Allowed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitDecision {
    Allowed,
    Limited { retry_after: u64 },
}

impl AppMetrics {
    pub fn record_http_request(&self, elapsed_ms: u128, failed: bool) {
        self.http_requests_total.fetch_add(1, Ordering::Relaxed);
        if failed {
            self.http_requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.total_request_ms.fetch_add(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_job_started(&self) {
        self.jobs_started.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_job_succeeded(&self, elapsed_ms: u128) {
        self.jobs_succeeded.fetch_add(1, Ordering::Relaxed);
        self.total_inference_ms.fetch_add(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_job_failed(&self, elapsed_ms: u128) {
        self.jobs_failed.fetch_add(1, Ordering::Relaxed);
        self.total_inference_ms.fetch_add(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_job_timed_out(&self, elapsed_ms: u128) {
        self.jobs_timed_out.fetch_add(1, Ordering::Relaxed);
        self.record_job_failed(elapsed_ms);
    }

    pub fn record_worker_restart(&self) {
        self.worker_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_model_load(&self, elapsed_ms: u128) {
        self.model_load_ms.store(
            usize::try_from(elapsed_ms).unwrap_or(usize::MAX),
            Ordering::Relaxed,
        );
    }

    pub fn record_webhook_failure(&self) {
        self.webhook_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cleanup_failure(&self) {
        self.cleanup_failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let model_load_ms = self.model_load_ms.load(Ordering::Relaxed);
        MetricsSnapshot {
            jobs_started: self.jobs_started.load(Ordering::Relaxed),
            jobs_succeeded: self.jobs_succeeded.load(Ordering::Relaxed),
            jobs_failed: self.jobs_failed.load(Ordering::Relaxed),
            jobs_timed_out: self.jobs_timed_out.load(Ordering::Relaxed),
            worker_restarts: self.worker_restarts.load(Ordering::Relaxed),
            total_inference_ms: self.total_inference_ms.load(Ordering::Relaxed),
            model_load_ms: (model_load_ms > 0).then_some(model_load_ms),
            http_requests_total: self.http_requests_total.load(Ordering::Relaxed),
            http_requests_failed: self.http_requests_failed.load(Ordering::Relaxed),
            total_request_ms: self.total_request_ms.load(Ordering::Relaxed),
            webhook_failures: self.webhook_failures.load(Ordering::Relaxed),
            cleanup_failures: self.cleanup_failures.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limiter_is_disabled_with_zero_limit() {
        let limiter = RateLimiter::new(0);

        for _ in 0..10 {
            assert_eq!(limiter.check(), RateLimitDecision::Allowed);
        }
    }

    #[test]
    fn rate_limiter_rejects_after_window_limit() {
        let limiter = RateLimiter::new(1);

        assert_eq!(limiter.check(), RateLimitDecision::Allowed);
        assert!(matches!(
            limiter.check(),
            RateLimitDecision::Limited {
                retry_after: 1..=60
            }
        ));
    }
}
