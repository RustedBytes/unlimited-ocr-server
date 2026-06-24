use std::{env, path::PathBuf};

use anyhow::anyhow;
use serde::{Deserialize, Serialize};

use super::defaults::DEFAULT_MODEL_PATH;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelVariant {
    UnlimitedOcr,
    Custom,
}

impl ModelVariant {
    pub fn default_model_path(self) -> PathBuf {
        PathBuf::from(DEFAULT_MODEL_PATH)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnlimitedOcr => "unlimited_ocr",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) enum ModelPathSelection {
    ExplicitPath,
    VariantDefaultPath,
}

pub(super) fn parse_model_variant(
    file_value: Option<String>,
    model_path_selection: ModelPathSelection,
) -> anyhow::Result<ModelVariant> {
    let Some(raw) = env::var("MODEL_VARIANT").ok().or(file_value) else {
        return Ok(match model_path_selection {
            ModelPathSelection::ExplicitPath => ModelVariant::Custom,
            ModelPathSelection::VariantDefaultPath => ModelVariant::UnlimitedOcr,
        });
    };

    parse_model_variant_value(&raw)
}

pub(super) fn parse_model_variant_value(raw: &str) -> anyhow::Result<ModelVariant> {
    let normalized = raw.trim().to_ascii_lowercase().replace(['-', '_'], "");
    match normalized.as_str() {
        "unlimitedocr" | "ocr" | "default" => Ok(ModelVariant::UnlimitedOcr),
        "custom" => Ok(ModelVariant::Custom),
        _ => Err(anyhow!(
            "unsupported model variant `{raw}`; expected one of unlimited_ocr, custom"
        )),
    }
}
