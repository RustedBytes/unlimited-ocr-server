use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::ModelVariant;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrResult {
    pub text: String,
    pub detections: Vec<OcrDetection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tables: Vec<OcrTable>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrDetection {
    pub label: String,
    pub bbox: BoundingBox,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrTable {
    pub bbox: BoundingBox,
    pub html: String,
    pub rows: Vec<Vec<OcrTableCell>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrTableCell {
    pub text: String,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub row_span: usize,
    #[serde(default = "one", skip_serializing_if = "is_one")]
    pub col_span: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x_min: i64,
    pub y_min: i64,
    pub x_max: i64,
    pub y_max: i64,
}

fn one() -> usize {
    1
}

fn is_one(value: &usize) -> bool {
    *value == 1
}

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
