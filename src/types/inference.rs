use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ModelVariant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    pub backend: String,
    pub model_path: PathBuf,
    pub model_variant: ModelVariant,
    pub input_name: String,
    pub input_dtype: String,
    pub original_width: u32,
    pub original_height: u32,
    pub processed_width: u32,
    pub processed_height: u32,
    pub elapsed_ms: u128,
    pub task_token: String,
    pub prompt_text: String,
    pub generated_text: String,
    pub generated_tokens: usize,
    pub result: Value,
    pub generations: Vec<GenerationMetadata>,
    pub outputs: Vec<TensorMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub task_token: String,
    pub prompt_text: String,
    pub generated_text: String,
    pub generated_tokens: usize,
    pub result: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub shape: Vec<i64>,
    pub elements: usize,
    pub mean: Option<f32>,
    pub min: Option<f32>,
    pub max: Option<f32>,
}
