use anyhow::anyhow;
use half::{bf16, f16};
use ort::value::{DynValue, Shape, TensorElementType, ValueType};

use crate::types::TensorMetadata;

#[derive(Debug, Clone)]
pub(super) struct TensorData {
    pub(super) shape: Vec<i64>,
    pub(super) data: Vec<f32>,
}

pub(super) fn extract_output_tensor(value: &DynValue, name: &str) -> anyhow::Result<TensorData> {
    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow!("output `{name}` is not an f32 tensor: {err}"))?;
            Ok(TensorData {
                shape: shape.iter().copied().collect(),
                data: data.to_vec(),
            })
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|err| anyhow!("output `{name}` is not an f16 tensor: {err}"))?;
            Ok(TensorData {
                shape: shape.iter().copied().collect(),
                data: data.iter().map(|value| value.to_f32()).collect(),
            })
        }
        ValueType::Tensor {
            ty: TensorElementType::Bfloat16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<bf16>()
                .map_err(|err| anyhow!("output `{name}` is not a bf16 tensor: {err}"))?;
            Ok(TensorData {
                shape: shape.iter().copied().collect(),
                data: data.iter().map(|value| value.to_f32()).collect(),
            })
        }
        other => Err(anyhow!("output `{name}` has unsupported dtype: {other:?}")),
    }
}

pub(super) fn argmax_token_at_position(
    logits: &TensorData,
    position: usize,
) -> anyhow::Result<i64> {
    if logits.shape.len() != 3 {
        return Err(anyhow!(
            "logits tensor has invalid shape {:?}",
            logits.shape
        ));
    }
    let seq_len = usize::try_from(logits.shape[1])
        .map_err(|_| anyhow!("logits sequence length is invalid: {}", logits.shape[1]))?;
    let vocab_size = usize::try_from(logits.shape[2])
        .map_err(|_| anyhow!("logits vocabulary size is invalid: {}", logits.shape[2]))?;
    if position >= seq_len || vocab_size == 0 {
        return Err(anyhow!(
            "logits tensor shape {:?} cannot select position {position}",
            logits.shape
        ));
    }
    let start = position
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;
    let end = start
        .checked_add(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;
    let row = logits
        .data
        .get(start..end)
        .ok_or_else(|| anyhow!("logits data length does not match shape {:?}", logits.shape))?;
    let (idx, _) = row
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| anyhow!("failed to select next token from logits"))?;
    Ok(idx as i64)
}

pub(super) fn tensor_metadata_f32(name: &str, shape: &Shape, data: &[f32]) -> TensorMetadata {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &value in data {
        min = min.min(value);
        max = max.max(value);
        sum += value as f64;
    }

    TensorMetadata {
        name: name.to_string(),
        shape: shape.iter().copied().collect(),
        elements: data.len(),
        mean: (!data.is_empty()).then_some((sum / data.len() as f64) as f32),
        min: (!data.is_empty()).then_some(min),
        max: (!data.is_empty()).then_some(max),
    }
}
