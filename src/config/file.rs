use std::{env, fs as std_fs, path::PathBuf};

use anyhow::{Context, anyhow};
use serde::Deserialize;

use super::{defaults::DEFAULT_CONFIG_PATH, settings::env_path};

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct FileConfig {
    pub(super) server: Option<ServerConfig>,
    pub(super) model: Option<ModelConfig>,
    pub(super) queue: Option<QueueConfig>,
    pub(super) generation: Option<GenerationConfig>,
    pub(super) runtime: Option<RuntimeConfig>,
    pub(super) logging: Option<LoggingConfig>,
    pub(super) retention: Option<RetentionConfig>,
    pub(super) validation: Option<ValidationConfig>,
}

impl FileConfig {
    pub(super) fn load(config_path: Option<PathBuf>) -> anyhow::Result<Self> {
        // CLI path wins over CONFIG_PATH. Missing default config.toml is OK so
        // the binary can still run with built-in defaults.
        let has_cli_path = config_path.is_some();
        let has_env_path = env::var_os("CONFIG_PATH").is_some();
        let config_path = config_path
            .or_else(|| env_path("CONFIG_PATH"))
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));
        let has_explicit_path = has_cli_path || has_env_path;

        if !config_path.exists() {
            if has_explicit_path {
                return Err(anyhow!(
                    "config file does not exist: {}",
                    config_path.display()
                ));
            }

            return Ok(Self::default());
        }

        let contents = std_fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config file {}", config_path.display()))?;
        toml::from_str(&contents)
            .with_context(|| format!("failed to parse TOML config {}", config_path.display()))
    }
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct ServerConfig {
    pub(super) bind_addr: Option<String>,
    pub(super) data_dir: Option<PathBuf>,
    pub(super) allow_local_paths: Option<bool>,
    pub(super) local_path_roots: Option<Vec<PathBuf>>,
    pub(super) cors_allowed_origins: Option<Vec<String>>,
    pub(super) api_keys: Option<Vec<String>>,
    pub(super) rate_limit_requests_per_minute: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct ModelConfig {
    pub(super) variant: Option<String>,
    pub(super) path: Option<PathBuf>,
    pub(super) decode_path: Option<PathBuf>,
    pub(super) image_size: Option<u32>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct QueueConfig {
    pub(super) model_pool_size: Option<usize>,
    pub(super) queue_size: Option<usize>,
    pub(super) body_limit_bytes: Option<usize>,
    pub(super) request_timeout_seconds: Option<u64>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct GenerationConfig {
    pub(super) max_new_tokens: Option<usize>,
    pub(super) job_timeout_seconds: Option<u64>,
    pub(super) webhook_timeout_seconds: Option<u64>,
    pub(super) webhook_connect_timeout_seconds: Option<u64>,
    pub(super) webhook_max_attempts: Option<usize>,
    pub(super) webhook_initial_backoff_ms: Option<u64>,
    pub(super) webhook_signing_secret: Option<String>,
    pub(super) allow_private_webhook_urls: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct RuntimeConfig {
    pub(super) execution_providers: Option<Vec<String>>,
    pub(super) device_id: Option<i32>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct LoggingConfig {
    pub(super) rust_log: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct RetentionConfig {
    pub(super) job_retention_limit: Option<usize>,
    pub(super) metadata_retention_limit: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
pub(super) struct ValidationConfig {
    pub(super) max_image_width: Option<u32>,
    pub(super) max_image_height: Option<u32>,
    pub(super) max_pdf_pages: Option<usize>,
    pub(super) pdf_render_dpi: Option<u32>,
}
