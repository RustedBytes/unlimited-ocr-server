mod defaults;
mod file;
mod model_variant;
mod settings;

use std::{net::SocketAddr, path::PathBuf};

use anyhow::{Context, anyhow};
use log::debug;
use tokio::fs;

use self::{
    defaults::{
        DEFAULT_ADDR, DEFAULT_BODY_LIMIT_BYTES, DEFAULT_DATA_DIR, DEFAULT_JOB_RETENTION_LIMIT,
        DEFAULT_JOB_TIMEOUT_SECONDS, DEFAULT_MAX_IMAGE_HEIGHT, DEFAULT_MAX_IMAGE_WIDTH,
        DEFAULT_MAX_NEW_TOKENS, DEFAULT_METADATA_RETENTION_LIMIT, DEFAULT_MODEL_IMAGE_SIZE,
        DEFAULT_QUEUE_SIZE, DEFAULT_REQUEST_TIMEOUT_SECONDS, DEFAULT_RUST_LOG,
        DEFAULT_WEBHOOK_CONNECT_TIMEOUT_SECONDS, DEFAULT_WEBHOOK_INITIAL_BACKOFF_MS,
        DEFAULT_WEBHOOK_MAX_ATTEMPTS, DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
    },
    file::FileConfig,
    model_variant::{ModelPathSelection, parse_model_variant},
    settings::{
        SettingSource, bool_setting, env_path, execution_providers_setting, path_list_setting,
        path_setting, secret_setting, string_list_setting, string_setting, u32_setting,
        u64_setting, usize_setting,
    },
};

pub use self::model_variant::ModelVariant;

pub struct Config {
    pub addr: SocketAddr,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
    pub model_image_size: u32,
    pub data_dir: PathBuf,
    pub images_dir: PathBuf,
    pub metadata_dir: PathBuf,
    pub submissions_jsonl: PathBuf,
    pub results_jsonl: PathBuf,
    pub allow_local_paths: bool,
    pub local_path_roots: Vec<PathBuf>,
    pub cors_allowed_origins: Vec<String>,
    pub api_keys: Vec<String>,
    pub rate_limit_requests_per_minute: u64,
    pub job_retention_limit: usize,
    pub metadata_retention_limit: usize,
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub workers: usize,
    pub queue_size: usize,
    pub body_limit_bytes: usize,
    pub rust_log: String,
    pub max_new_tokens: usize,
    pub job_timeout_seconds: u64,
    pub request_timeout_seconds: u64,
    pub webhook_timeout_seconds: u64,
    pub webhook_connect_timeout_seconds: u64,
    pub webhook_max_attempts: usize,
    pub webhook_initial_backoff_ms: u64,
    pub webhook_signing_secret: Option<String>,
    pub webhooks_dead_letter_jsonl: PathBuf,
    pub allow_private_webhook_urls: bool,
    pub execution_providers: Vec<String>,
}

impl Config {
    pub fn load(config_path: Option<PathBuf>) -> anyhow::Result<Self> {
        let file_config = FileConfig::load(config_path)?;
        let server = file_config.server.unwrap_or_default();
        let model = file_config.model.unwrap_or_default();
        let queue = file_config.queue.unwrap_or_default();
        let generation = file_config.generation.unwrap_or_default();
        let runtime = file_config.runtime.unwrap_or_default();
        let logging = file_config.logging.unwrap_or_default();
        let retention = file_config.retention.unwrap_or_default();
        let validation = file_config.validation.unwrap_or_default();
        let execution_providers = execution_providers_setting(runtime.execution_providers);

        let data_dir = path_setting(
            SettingSource::new("DATA_DIR", server.data_dir),
            DEFAULT_DATA_DIR,
        );
        let metadata_dir = data_dir.join("metadata");
        let webhooks_dead_letter_jsonl = metadata_dir.join("webhooks_dead_letter.jsonl");
        let allow_local_paths = bool_setting(
            SettingSource::new("ALLOW_LOCAL_PATHS", server.allow_local_paths),
            false,
        )?;
        let local_path_roots = path_list_setting("LOCAL_PATH_ROOTS", server.local_path_roots);
        if allow_local_paths && local_path_roots.is_empty() {
            return Err(anyhow!(
                "local path inference requires at least one configured local_path_roots entry"
            ));
        }
        let cors_allowed_origins =
            string_list_setting("CORS_ALLOWED_ORIGINS", server.cors_allowed_origins);
        let api_keys = string_list_setting("API_KEYS", server.api_keys);
        let model_path_override = env_path("MODEL_PATH").or(model.path);
        let model_path_selection = if model_path_override.is_some() {
            ModelPathSelection::ExplicitPath
        } else {
            ModelPathSelection::VariantDefaultPath
        };
        let model_variant = parse_model_variant(model.variant, model_path_selection)?;
        let model_path = model_path_override.unwrap_or_else(|| model_variant.default_model_path());
        let addr = string_setting(
            SettingSource::new("BIND_ADDR", server.bind_addr),
            DEFAULT_ADDR,
        )
        .parse()
        .context("BIND_ADDR must be a socket address, for example 127.0.0.1:3000")?;

        Ok(Self {
            addr,
            model_path,
            model_variant,
            model_image_size: u32_setting(
                SettingSource::new("MODEL_IMAGE_SIZE", model.image_size),
                DEFAULT_MODEL_IMAGE_SIZE,
            )?
            .max(1),
            images_dir: data_dir.join("images"),
            submissions_jsonl: metadata_dir.join("submissions.jsonl"),
            results_jsonl: metadata_dir.join("results.jsonl"),
            allow_local_paths,
            local_path_roots,
            cors_allowed_origins,
            api_keys,
            rate_limit_requests_per_minute: u64_setting(
                SettingSource::new(
                    "RATE_LIMIT_REQUESTS_PER_MINUTE",
                    server.rate_limit_requests_per_minute,
                ),
                0,
            )?,
            job_retention_limit: usize_setting(
                SettingSource::new("JOB_RETENTION_LIMIT", retention.job_retention_limit),
                DEFAULT_JOB_RETENTION_LIMIT,
            )?,
            metadata_retention_limit: usize_setting(
                SettingSource::new(
                    "METADATA_RETENTION_LIMIT",
                    retention.metadata_retention_limit,
                ),
                DEFAULT_METADATA_RETENTION_LIMIT,
            )?,
            max_image_width: u32_setting(
                SettingSource::new("MAX_IMAGE_WIDTH", validation.max_image_width),
                DEFAULT_MAX_IMAGE_WIDTH,
            )?,
            max_image_height: u32_setting(
                SettingSource::new("MAX_IMAGE_HEIGHT", validation.max_image_height),
                DEFAULT_MAX_IMAGE_HEIGHT,
            )?,
            metadata_dir,
            data_dir,
            workers: usize_setting(
                SettingSource::new("MODEL_POOL_SIZE", queue.model_pool_size),
                default_worker_count_for_execution_providers(&execution_providers),
            )?
            .max(1),
            queue_size: usize_setting(
                SettingSource::new("QUEUE_SIZE", queue.queue_size),
                DEFAULT_QUEUE_SIZE,
            )?,
            body_limit_bytes: usize_setting(
                SettingSource::new("BODY_LIMIT_BYTES", queue.body_limit_bytes),
                DEFAULT_BODY_LIMIT_BYTES,
            )?,
            request_timeout_seconds: u64_setting(
                SettingSource::new("REQUEST_TIMEOUT_SECONDS", queue.request_timeout_seconds),
                DEFAULT_REQUEST_TIMEOUT_SECONDS,
            )?,
            rust_log: string_setting(
                SettingSource::new("RUST_LOG", logging.rust_log),
                DEFAULT_RUST_LOG,
            ),
            max_new_tokens: usize_setting(
                SettingSource::new("MAX_NEW_TOKENS", generation.max_new_tokens),
                DEFAULT_MAX_NEW_TOKENS,
            )?
            .max(1),
            job_timeout_seconds: u64_setting(
                SettingSource::new("JOB_TIMEOUT_SECONDS", generation.job_timeout_seconds),
                DEFAULT_JOB_TIMEOUT_SECONDS,
            )?,
            webhook_timeout_seconds: u64_setting(
                SettingSource::new(
                    "WEBHOOK_TIMEOUT_SECONDS",
                    generation.webhook_timeout_seconds,
                ),
                DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
            )?,
            webhook_connect_timeout_seconds: u64_setting(
                SettingSource::new(
                    "WEBHOOK_CONNECT_TIMEOUT_SECONDS",
                    generation.webhook_connect_timeout_seconds,
                ),
                DEFAULT_WEBHOOK_CONNECT_TIMEOUT_SECONDS,
            )?,
            webhook_max_attempts: usize_setting(
                SettingSource::new("WEBHOOK_MAX_ATTEMPTS", generation.webhook_max_attempts),
                DEFAULT_WEBHOOK_MAX_ATTEMPTS,
            )?
            .max(1),
            webhook_initial_backoff_ms: u64_setting(
                SettingSource::new(
                    "WEBHOOK_INITIAL_BACKOFF_MS",
                    generation.webhook_initial_backoff_ms,
                ),
                DEFAULT_WEBHOOK_INITIAL_BACKOFF_MS,
            )?,
            webhook_signing_secret: secret_setting(
                "WEBHOOK_SIGNING_SECRET",
                generation.webhook_signing_secret,
            ),
            webhooks_dead_letter_jsonl,
            allow_private_webhook_urls: bool_setting(
                SettingSource::new(
                    "ALLOW_PRIVATE_WEBHOOK_URLS",
                    generation.allow_private_webhook_urls,
                ),
                false,
            )?,
            execution_providers,
        })
    }

    pub async fn ensure_dirs(&self) -> anyhow::Result<()> {
        if !self.model_path.exists() {
            return Err(anyhow!(
                "model file does not exist: {}",
                self.model_path.display()
            ));
        }
        debug!(
            "ensuring data directories images_dir={} metadata_dir={}",
            self.images_dir.display(),
            self.metadata_dir.display()
        );
        fs::create_dir_all(&self.images_dir).await?;
        fs::create_dir_all(&self.metadata_dir).await?;
        Ok(())
    }
}

fn default_worker_count() -> usize {
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .clamp(1, 4)
}

fn default_worker_count_for_execution_providers(execution_providers: &[String]) -> usize {
    if execution_providers
        .iter()
        .any(|provider| provider.as_str() == "cuda")
    {
        1
    } else {
        default_worker_count()
    }
}

#[cfg(test)]
mod tests;
