use std::{
    collections::HashMap,
    io::ErrorKind,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::anyhow;
use async_channel::Receiver;
use log::{debug, error, info, warn};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::{
    config::Config,
    inference::UnlimitedOcrWorker,
    state::{AppMetrics, WorkerPoolState},
    types::{InferenceMetadata, JobRecord, JobStatus},
    util::append_jsonl,
};

use super::{WebhookClient, queue::JobRequest, store::retain_recent_jobs, webhook::send_webhook};

pub fn start_workers(
    config: Arc<Config>,
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    workers: Arc<WorkerPoolState>,
    metrics: Arc<AppMetrics>,
    webhooks: Arc<WebhookClient>,
    queue_rx: Receiver<JobRequest>,
) {
    info!("starting model worker pool workers={}", config.workers);
    for worker_id in 0..config.workers {
        debug!("spawning model worker worker_id={}", worker_id);
        tokio::spawn(worker_loop(
            worker_id,
            WorkerRuntime {
                config: Arc::clone(&config),
                jobs: Arc::clone(&jobs),
                workers: Arc::clone(&workers),
                metrics: Arc::clone(&metrics),
                webhooks: Arc::clone(&webhooks),
                queue_rx: queue_rx.clone(),
            },
        ));
    }
}

struct WorkerRuntime {
    config: Arc<Config>,
    jobs: Arc<RwLock<HashMap<Uuid, JobRecord>>>,
    workers: Arc<WorkerPoolState>,
    metrics: Arc<AppMetrics>,
    webhooks: Arc<WebhookClient>,
    queue_rx: Receiver<JobRequest>,
}

async fn worker_loop(worker_id: usize, runtime: WorkerRuntime) {
    loop {
        let Some(mut worker) = initialize_worker(
            worker_id,
            Arc::clone(&runtime.config),
            &runtime.workers,
            &runtime.metrics,
        )
        .await
        else {
            return;
        };

        let should_restart = process_worker_queue(worker_id, &runtime, &mut worker).await;

        runtime.workers.mark_stopped();
        if !should_restart {
            return;
        }
    }
}

async fn process_worker_queue(
    worker_id: usize,
    runtime: &WorkerRuntime,
    worker: &mut UnlimitedOcrWorker,
) -> bool {
    while let Ok(request) = runtime.queue_rx.recv().await {
        if process_worker_request(worker_id, runtime, request, worker).await {
            return true;
        }
    }

    false
}

async fn process_worker_request(
    worker_id: usize,
    runtime: &WorkerRuntime,
    request: JobRequest,
    worker: &mut UnlimitedOcrWorker,
) -> bool {
    let job_span = tracing::info_span!(
        "inference_job",
        job_id = %request.id,
        worker_id,
        text_input_present = request.task.text_input.is_some(),
    );
    job_span.in_scope(|| {
        debug!(
            "worker received job worker_id={} job_id={} text_input_present={} image_path={}",
            worker_id,
            request.id,
            request.task.text_input.is_some(),
            request.image_path.display()
        );
    });
    runtime.metrics.record_job_started();
    let job_started = Instant::now();
    mark_running(&runtime.jobs, request.id).await;

    let join_handle = spawn_inference(worker_id, worker, &request, job_span);
    let result = wait_for_inference(&runtime.config, join_handle).await;
    let elapsed_ms = job_started.elapsed().as_millis();

    handle_inference_result(runtime, request.id, elapsed_ms, result, worker).await
}

fn spawn_inference(
    worker_id: usize,
    worker: &mut UnlimitedOcrWorker,
    request: &JobRequest,
    job_span: tracing::Span,
) -> BlockingInferenceJoin {
    tokio::task::spawn_blocking({
        let image_path = request.image_path.clone();
        let task = request.task.clone();
        let mut worker = worker.take_for_blocking();
        move || {
            job_span.in_scope(|| {
                let result = worker.infer(&image_path, &task);
                (worker_id, worker, result)
            })
        }
    })
}

async fn handle_inference_result(
    runtime: &WorkerRuntime,
    job_id: Uuid,
    elapsed_ms: u128,
    result: InferenceRunResult,
    worker: &mut UnlimitedOcrWorker,
) -> bool {
    match result {
        InferenceRunResult::Completed(result) => {
            handle_completed_inference(runtime, job_id, elapsed_ms, *result, worker).await
        }
        InferenceRunResult::TimedOut => {
            runtime.metrics.record_job_timed_out(elapsed_ms);
            runtime.metrics.record_worker_restart();
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(anyhow!(
                    "job timed out after {} seconds",
                    runtime.config.job_timeout_seconds
                )),
            )
            .await;
            true
        }
    }
}

async fn handle_completed_inference(
    runtime: &WorkerRuntime,
    job_id: Uuid,
    elapsed_ms: u128,
    result: BlockingInferenceResult,
    worker: &mut UnlimitedOcrWorker,
) -> bool {
    match result {
        Ok((_, returned_worker, Ok(metadata))) => {
            *worker = returned_worker;
            runtime.metrics.record_job_succeeded(elapsed_ms);
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Ok(metadata),
            )
            .await;
            false
        }
        Ok((_, returned_worker, Err(err))) => {
            *worker = returned_worker;
            runtime.metrics.record_job_failed(elapsed_ms);
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(err),
            )
            .await;
            false
        }
        Err(err) => {
            runtime.metrics.record_job_failed(elapsed_ms);
            runtime.metrics.record_worker_restart();
            finish_job(
                &runtime.config,
                &runtime.jobs,
                &runtime.metrics,
                &runtime.webhooks,
                job_id,
                Err(anyhow!("worker task failed: {err}")),
            )
            .await;
            true
        }
    }
}

async fn initialize_worker(
    worker_id: usize,
    config: Arc<Config>,
    workers: &WorkerPoolState,
    metrics: &AppMetrics,
) -> Option<UnlimitedOcrWorker> {
    let started = Instant::now();
    let worker = tokio::task::spawn_blocking({
        let config = Arc::clone(&config);
        move || UnlimitedOcrWorker::new(worker_id, config)
    })
    .await;

    match worker {
        Ok(Ok(worker)) => {
            metrics.record_model_load(started.elapsed().as_millis());
            workers.mark_ready();
            info!("model worker ready worker_id={}", worker_id);
            Some(worker)
        }
        Ok(Err(err)) => {
            workers.mark_failed();
            error!(
                "failed to initialize model worker worker_id={} error={}",
                worker_id, err
            );
            None
        }
        Err(err) => {
            workers.mark_failed();
            error!(
                "model worker initialization panicked worker_id={} error={}",
                worker_id, err
            );
            None
        }
    }
}

type BlockingInferenceJoin =
    tokio::task::JoinHandle<(usize, UnlimitedOcrWorker, anyhow::Result<InferenceMetadata>)>;
type BlockingInferenceResult =
    Result<(usize, UnlimitedOcrWorker, anyhow::Result<InferenceMetadata>), tokio::task::JoinError>;

enum InferenceRunResult {
    Completed(Box<BlockingInferenceResult>),
    TimedOut,
}

async fn wait_for_inference(
    config: &Config,
    join_handle: BlockingInferenceJoin,
) -> InferenceRunResult {
    if config.job_timeout_seconds == 0 {
        return InferenceRunResult::Completed(Box::new(join_handle.await));
    }

    match tokio::time::timeout(Duration::from_secs(config.job_timeout_seconds), join_handle).await {
        Ok(result) => InferenceRunResult::Completed(Box::new(result)),
        Err(_) => InferenceRunResult::TimedOut,
    }
}

async fn mark_running(jobs: &RwLock<HashMap<Uuid, JobRecord>>, id: Uuid) {
    let mut jobs = jobs.write().await;
    if let Some(record) = jobs.get_mut(&id) {
        record.status = JobStatus::Running;
        record.updated_at = time::OffsetDateTime::now_utc();
        info!(
            "job running job_id={} image_path={}",
            id,
            record.image_path.display()
        );
    } else {
        warn!("job missing while marking running job_id={}", id);
    }
}

pub(super) async fn finish_job(
    config: &Config,
    jobs: &RwLock<HashMap<Uuid, JobRecord>>,
    metrics: &AppMetrics,
    webhooks: &WebhookClient,
    id: Uuid,
    result: anyhow::Result<InferenceMetadata>,
) {
    let mut final_record = None;
    {
        let mut jobs = jobs.write().await;
        if let Some(record) = jobs.get_mut(&id) {
            record.updated_at = time::OffsetDateTime::now_utc();
            match result {
                Ok(metadata) => {
                    record.status = JobStatus::Succeeded;
                    record.result = Some(metadata);
                    record.error = None;
                    info!("job succeeded job_id={}", id);
                }
                Err(err) => {
                    record.status = JobStatus::Failed;
                    record.error = Some(err.to_string());
                    warn!("job failed job_id={} error={}", id, err);
                }
            }
            final_record = Some(record.clone());
        }
    }

    if let Some(record) = final_record {
        if let Err(err) = append_jsonl(&config.results_jsonl, &record).await {
            error!(
                "failed to append result metadata job_id={} path={} error={}",
                id,
                config.results_jsonl.display(),
                err
            );
        } else {
            debug!(
                "result metadata appended job_id={} path={}",
                id,
                config.results_jsonl.display()
            );
        }
        send_webhook(metrics, webhooks, &record).await;
        cleanup_job_artifacts(config, metrics, &record).await;
        {
            let mut jobs = jobs.write().await;
            retain_recent_jobs(&mut jobs, config.job_retention_limit);
        }
    } else {
        warn!("job missing while finishing job_id={}", id);
    }
}

async fn cleanup_job_artifacts(config: &Config, metrics: &AppMetrics, record: &JobRecord) {
    if !should_delete_image(config, record) {
        return;
    }

    match tokio::fs::remove_file(&record.image_path).await {
        Ok(()) => {
            info!(
                "uploaded image cleaned up job_id={} path={}",
                record.id,
                record.image_path.display()
            );
        }
        Err(err) if err.kind() == ErrorKind::NotFound => {
            debug!(
                "uploaded image already removed job_id={} path={}",
                record.id,
                record.image_path.display()
            );
        }
        Err(err) => {
            metrics.record_cleanup_failure();
            warn!(
                "failed to clean up uploaded image job_id={} path={} error={}",
                record.id,
                record.image_path.display(),
                err
            );
        }
    }
}

pub(super) fn should_delete_image(config: &Config, record: &JobRecord) -> bool {
    matches!(
        record.input_kind.as_str(),
        "upload" | "upload_pdf_page" | "local_pdf_page"
    ) && path_is_inside(&record.image_path, &config.images_dir)
}

fn path_is_inside(path: &Path, directory: &Path) -> bool {
    path.starts_with(directory) && path != directory
}
