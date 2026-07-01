mod api;
mod config;
mod inference;
mod jobs;
mod pdf;
mod state;
mod templates;
mod types;
mod util;

use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, anyhow};
use clap::Parser;
use log::{debug, info};
use state::{AppMetrics, AppState, RateLimiter, WorkerPoolState};
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

use crate::{
    config::Config,
    inference::validate_model_artifacts,
    jobs::{JobRequest, WebhookClient, load_jobs, start_workers},
};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    /// TOML config file path. Overrides CONFIG_PATH when set.
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = Arc::new(Config::load(cli.config)?);

    init_tracing(&config.rust_log)?;
    prepare_config(&config).await?;

    let runtime = build_runtime_parts(&config).await?;
    start_workers(
        Arc::clone(&config),
        Arc::clone(&runtime.state.jobs),
        runtime.workers,
        runtime.metrics,
        runtime.webhooks,
        runtime.queue_rx,
    );

    serve(config, runtime.state).await
}

struct RuntimeParts {
    state: AppState,
    queue_rx: async_channel::Receiver<JobRequest>,
    workers: Arc<WorkerPoolState>,
    metrics: Arc<AppMetrics>,
    webhooks: Arc<WebhookClient>,
}

async fn prepare_config(config: &Config) -> anyhow::Result<()> {
    config.ensure_dirs().await?;
    validate_model_artifacts(&config.model_path, config.model_variant)?;
    log_loaded_config(config);
    Ok(())
}

async fn build_runtime_parts(config: &Arc<Config>) -> anyhow::Result<RuntimeParts> {
    let (queue_tx, queue_rx) = async_channel::bounded(config.queue_size);
    let jobs = load_jobs(config).await?;
    let workers = Arc::new(WorkerPoolState::new(config.workers));
    let metrics = Arc::new(AppMetrics::default());
    let webhooks = Arc::new(WebhookClient::from_config(config)?);
    let state = AppState {
        config: Arc::clone(config),
        queue_tx,
        jobs: Arc::new(RwLock::new(jobs)),
        workers: Arc::clone(&workers),
        metrics: Arc::clone(&metrics),
        rate_limiter: Arc::new(RateLimiter::new(config.rate_limit_requests_per_minute)),
    };
    Ok(RuntimeParts {
        state,
        queue_rx,
        workers,
        metrics,
        webhooks,
    })
}

async fn serve(config: Arc<Config>, state: AppState) -> anyhow::Result<()> {
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(config.addr)
        .await
        .context("failed to bind TCP listener")?;

    log_server_listening(&config);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

fn log_loaded_config(config: &Config) {
    debug!(
        "config loaded addr={} model_path={} decode_model_path={} model_variant={} model_image_size={} data_dir={} images_dir={} metadata_dir={} workers={} queue_size={} body_limit_bytes={} request_timeout_seconds={} max_pdf_pages={} pdf_render_dpi={} api_key_auth_enabled={} rate_limit_requests_per_minute={} max_new_tokens={} temperature={} job_timeout_seconds={} webhook_timeout_seconds={} webhook_connect_timeout_seconds={} webhook_max_attempts={} webhook_initial_backoff_ms={} webhook_signing_enabled={} allow_private_webhook_urls={} execution_providers={:?} rust_log={}",
        config.addr,
        config.model_path.display(),
        config
            .decode_model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        config.model_variant.as_str(),
        config.model_image_size,
        config.data_dir.display(),
        config.images_dir.display(),
        config.metadata_dir.display(),
        config.workers,
        config.queue_size,
        config.body_limit_bytes,
        config.request_timeout_seconds,
        config.max_pdf_pages,
        config.pdf_render_dpi,
        !config.api_keys.is_empty(),
        config.rate_limit_requests_per_minute,
        config.max_new_tokens,
        config.temperature,
        config.job_timeout_seconds,
        config.webhook_timeout_seconds,
        config.webhook_connect_timeout_seconds,
        config.webhook_max_attempts,
        config.webhook_initial_backoff_ms,
        config.webhook_signing_secret.is_some(),
        config.allow_private_webhook_urls,
        config.execution_providers,
        config.rust_log
    );
}

fn log_server_listening(config: &Config) {
    info!(
        "server listening addr={} workers={} model={} decode_model={} model_variant={} model_image_size={} data_dir={} queue_size={} body_limit_bytes={} request_timeout_seconds={} max_pdf_pages={} pdf_render_dpi={} api_key_auth_enabled={} rate_limit_requests_per_minute={} max_new_tokens={} temperature={} job_timeout_seconds={} webhook_timeout_seconds={} webhook_connect_timeout_seconds={} webhook_max_attempts={} webhook_initial_backoff_ms={} webhook_signing_enabled={} allow_private_webhook_urls={} execution_providers={:?}",
        config.addr,
        config.workers,
        config.model_path.display(),
        config
            .decode_model_path
            .as_ref()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        config.model_variant.as_str(),
        config.model_image_size,
        config.data_dir.display(),
        config.queue_size,
        config.body_limit_bytes,
        config.request_timeout_seconds,
        config.max_pdf_pages,
        config.pdf_render_dpi,
        !config.api_keys.is_empty(),
        config.rate_limit_requests_per_minute,
        config.max_new_tokens,
        config.temperature,
        config.job_timeout_seconds,
        config.webhook_timeout_seconds,
        config.webhook_connect_timeout_seconds,
        config.webhook_max_attempts,
        config.webhook_initial_backoff_ms,
        config.webhook_signing_secret.is_some(),
        config.allow_private_webhook_urls,
        config.execution_providers
    );
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        log::error!("failed to install ctrl-c handler error={}", err);
    }
}

fn init_tracing(rust_log: &str) -> anyhow::Result<()> {
    let filter = EnvFilter::try_from_default_env().or_else(|_| EnvFilter::try_new(rust_log))?;
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .compact()
        .try_init()
        .map_err(|err| anyhow!("failed to initialize tracing subscriber: {err}"))?;
    Ok(())
}
