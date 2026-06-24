use anyhow::anyhow;
use half::{bf16, f16};
use ort::value::{DynValue, Shape, TensorElementType, ValueType};

use crate::types::TensorMetadata;

pub(super) fn argmax_token_from_output_at_position(
    value: &DynValue,
    name: &str,
    position: usize,
) -> anyhow::Result<i64> {
    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow!("output `{name}` is not an f32 tensor: {err}"))?;
            argmax_token_at_position(shape, data, position, |value| *value)
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|err| anyhow!("output `{name}` is not an f16 tensor: {err}"))?;
            argmax_token_at_position(shape, data, position, |value| value.to_f32())
        }
        ValueType::Tensor {
            ty: TensorElementType::Bfloat16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<bf16>()
                .map_err(|err| anyhow!("output `{name}` is not a bf16 tensor: {err}"))?;
            argmax_token_at_position(shape, data, position, |value| value.to_f32())
        }
        other => Err(anyhow!("output `{name}` has unsupported dtype: {other:?}")),
    }
}

fn argmax_token_at_position<T>(
    shape: &Shape,
    data: &[T],
    position: usize,
    to_f32: impl Fn(&T) -> f32,
) -> anyhow::Result<i64> {
    let shape_values = shape.iter().copied().collect::<Vec<_>>();
    if shape_values.len() != 3 {
        return Err(anyhow!("logits tensor has invalid shape {shape_values:?}"));
    }

    let seq_len = usize::try_from(shape_values[1])
        .map_err(|_| anyhow!("logits sequence length is invalid: {}", shape_values[1]))?;
    let vocab_size = usize::try_from(shape_values[2])
        .map_err(|_| anyhow!("logits vocabulary size is invalid: {}", shape_values[2]))?;

    if position >= seq_len || vocab_size == 0 {
        return Err(anyhow!(
            "logits tensor shape {shape_values:?} cannot select position {position}"
        ));
    }

    let start = position
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;
    let end = start
        .checked_add(vocab_size)
        .ok_or_else(|| anyhow!("logits shape is too large"))?;

    let row = data
        .get(start..end)
        .ok_or_else(|| anyhow!("logits data length does not match shape {shape_values:?}"))?;
    let (idx, _) = row
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| to_f32(left).total_cmp(&to_f32(right)))
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

#[cfg(test)]
mod tests {
    use half::{bf16, f16};
    use ort::value::{DynValue, Tensor};

    use super::argmax_token_from_output_at_position;

    #[test]
    fn selects_token_from_f32_logits_row() {
        let value: DynValue = Tensor::<f32>::from_array((
            [1_usize, 2, 4],
            vec![0.0, 9.0, 1.0, 2.0, -1.0, 3.0, 12.0, 4.0],
        ))
        .unwrap()
        .into_dyn();

        let got = argmax_token_from_output_at_position(&value, "logits", 1).unwrap();

        assert_eq!(got, 2);
    }

    #[test]
    fn selects_token_from_f16_logits_row_without_materializing_tensor() {
        let value: DynValue = Tensor::<f16>::from_array((
            [1_usize, 1, 3],
            vec![f16::from_f32(-2.0), f16::from_f32(8.0), f16::from_f32(1.0)],
        ))
        .unwrap()
        .into_dyn();

        let got = argmax_token_from_output_at_position(&value, "logits", 0).unwrap();

        assert_eq!(got, 1);
    }

    #[test]
    fn selects_token_from_bf16_logits_row_without_materializing_tensor() {
        let value: DynValue = Tensor::<bf16>::from_array((
            [1_usize, 1, 3],
            vec![
                bf16::from_f32(-2.0),
                bf16::from_f32(8.0),
                bf16::from_f32(1.0),
            ],
        ))
        .unwrap()
        .into_dyn();

        let got = argmax_token_from_output_at_position(&value, "logits", 0).unwrap();

        assert_eq!(got, 1);
    }

    #[test]
    fn rejects_position_outside_sequence_length() {
        let value: DynValue = Tensor::<f32>::from_array(([1_usize, 1, 2], vec![1.0, 2.0]))
            .unwrap()
            .into_dyn();

        let err = argmax_token_from_output_at_position(&value, "logits", 1).unwrap_err();

        assert!(
            err.to_string()
                .contains("logits tensor shape [1, 1, 2] cannot select position 1")
        );
    }
}
