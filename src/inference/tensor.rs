use anyhow::anyhow;
use half::{bf16, f16};
use ort::value::{DynValue, Shape, TensorElementType, ValueType};
use rand::Rng;

use crate::types::TensorMetadata;

pub(super) fn sample_token_from_output_at_position(
    value: &DynValue,
    name: &str,
    position: usize,
    temperature: f32,
) -> anyhow::Result<i64> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(anyhow!(
            "temperature must be a finite number 0 or greater, got {temperature}"
        ));
    }

    let mut rng = rand::rng();
    let sample = rng.random::<f64>();

    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|err| anyhow!("output `{name}` is not an f32 tensor: {err}"))?;
            sample_token_at_position(shape, data, position, temperature, sample, |value| *value)
        }
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<f16>()
                .map_err(|err| anyhow!("output `{name}` is not an f16 tensor: {err}"))?;
            sample_token_at_position(shape, data, position, temperature, sample, |value| {
                value.to_f32()
            })
        }
        ValueType::Tensor {
            ty: TensorElementType::Bfloat16,
            ..
        } => {
            let (shape, data) = value
                .try_extract_tensor::<bf16>()
                .map_err(|err| anyhow!("output `{name}` is not a bf16 tensor: {err}"))?;
            sample_token_at_position(shape, data, position, temperature, sample, |value| {
                value.to_f32()
            })
        }
        other => Err(anyhow!("output `{name}` has unsupported dtype: {other:?}")),
    }
}

fn sample_token_at_position<T>(
    shape: &Shape,
    data: &[T],
    position: usize,
    temperature: f32,
    sample: f64,
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

    select_token_from_logits_row(row, temperature, sample, to_f32)
}

fn select_token_from_logits_row<T>(
    row: &[T],
    temperature: f32,
    sample: f64,
    to_f32: impl Fn(&T) -> f32,
) -> anyhow::Result<i64> {
    if temperature == 0.0 {
        return argmax_token_from_logits_row(row, to_f32);
    }

    let logits = row.iter().map(&to_f32).collect::<Vec<_>>();
    let positive_infinity_count = logits
        .iter()
        .filter(|value| **value == f32::INFINITY)
        .count();
    if positive_infinity_count > 0 {
        let selected = ((sample.clamp(0.0, 1.0) * positive_infinity_count as f64).floor() as usize)
            .min(positive_infinity_count - 1);
        let token = logits
            .iter()
            .enumerate()
            .filter(|(_, value)| **value == f32::INFINITY)
            .nth(selected)
            .map(|(idx, _)| idx)
            .ok_or_else(|| anyhow!("failed to select next token from infinite logits"))?;
        return Ok(token as i64);
    }

    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(f32::total_cmp)
        .ok_or_else(|| anyhow!("failed to select next token from logits"))?;

    let mut total_weight = 0.0_f64;
    for &logit in &logits {
        if logit.is_finite() {
            let weight = ((logit - max) / temperature).exp() as f64;
            if weight.is_finite() && weight > 0.0 {
                total_weight += weight;
            }
        }
    }

    if total_weight <= 0.0 || !total_weight.is_finite() {
        return Err(anyhow!("failed to select next token from logits"));
    }

    let mut remaining = sample.clamp(0.0, 1.0) * total_weight;
    let mut fallback = None;
    for (idx, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            continue;
        }

        let weight = ((logit - max) / temperature).exp() as f64;
        if !weight.is_finite() || weight <= 0.0 {
            continue;
        }

        fallback = Some(idx);
        if remaining < weight {
            return Ok(idx as i64);
        }
        remaining -= weight;
    }

    fallback
        .map(|idx| idx as i64)
        .ok_or_else(|| anyhow!("failed to select next token from logits"))
}

fn argmax_token_from_logits_row<T>(row: &[T], to_f32: impl Fn(&T) -> f32) -> anyhow::Result<i64> {
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

    use super::{sample_token_from_output_at_position, select_token_from_logits_row};

    #[test]
    fn selects_token_from_f32_logits_row() {
        let value: DynValue = Tensor::<f32>::from_array((
            [1_usize, 2, 4],
            vec![0.0, 9.0, 1.0, 2.0, -1.0, 3.0, 12.0, 4.0],
        ))
        .unwrap()
        .into_dyn();

        let got = sample_token_from_output_at_position(&value, "logits", 1, 0.0).unwrap();

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

        let got = sample_token_from_output_at_position(&value, "logits", 0, 0.0).unwrap();

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

        let got = sample_token_from_output_at_position(&value, "logits", 0, 0.0).unwrap();

        assert_eq!(got, 1);
    }

    #[test]
    fn zero_temperature_selects_argmax_token() {
        let row = [10.0, 1.0, 2.0];

        let got = select_token_from_logits_row(&row, 0.0, 0.99, |value| *value).unwrap();

        assert_eq!(got, 0);
    }

    #[test]
    fn positive_temperature_samples_from_softmax_buckets() {
        let row = [0.0, 0.0];

        let first = select_token_from_logits_row(&row, 1.0, 0.25, |value| *value).unwrap();
        let second = select_token_from_logits_row(&row, 1.0, 0.75, |value| *value).unwrap();

        assert_eq!(first, 0);
        assert_eq!(second, 1);
    }

    #[test]
    fn positive_temperature_samples_uniformly_from_infinite_logits() {
        let row = [f32::INFINITY, 1.0, f32::INFINITY];

        let got = select_token_from_logits_row(&row, 1.0, 0.75, |value| *value).unwrap();

        assert_eq!(got, 2);
    }

    #[test]
    fn rejects_invalid_temperature() {
        let value: DynValue = Tensor::<f32>::from_array(([1_usize, 1, 2], vec![1.0, 2.0]))
            .unwrap()
            .into_dyn();

        let err =
            sample_token_from_output_at_position(&value, "logits", 0, f32::INFINITY).unwrap_err();

        assert!(
            err.to_string()
                .contains("temperature must be a finite number 0 or greater")
        );
    }

    #[test]
    fn rejects_position_outside_sequence_length() {
        let value: DynValue = Tensor::<f32>::from_array(([1_usize, 1, 2], vec![1.0, 2.0]))
            .unwrap()
            .into_dyn();

        let err = sample_token_from_output_at_position(&value, "logits", 1, 0.0).unwrap_err();

        assert!(
            err.to_string()
                .contains("logits tensor shape [1, 1, 2] cannot select position 1")
        );
    }
}
