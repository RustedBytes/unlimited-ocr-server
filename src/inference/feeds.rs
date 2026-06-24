use std::{borrow::Cow, collections::HashSet};

use anyhow::anyhow;
use half::{bf16, f16};
use ort::{
    session::SessionOutputs,
    session::{Session, SessionInputValue},
    value::{DynValue, Shape, Tensor, TensorElementType, ValueType},
};

type FeedValues = Vec<(Cow<'static, str>, SessionInputValue<'static>)>;

#[derive(Debug, Clone)]
pub(super) struct InputMetadata {
    pub(super) names: HashSet<String>,
    pub(super) image_dtype: TensorElementType,
    pub(super) fixed_sequence_length: Option<usize>,
    pub(super) kv_cache: KvCacheMetadata,
    fixed_image_size: Option<u32>,
}

#[derive(Debug, Clone)]
pub(super) struct DecodeInputMetadata {
    pub(super) names: HashSet<String>,
    pub(super) kv_cache: KvCacheMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct KvCacheMetadata {
    past_inputs: Vec<String>,
    present_outputs: Vec<String>,
}

#[derive(Debug)]
pub(super) struct KvCache {
    values: Vec<DynValue>,
}

pub(super) struct FeedInputs<'a> {
    pub(super) input_ids: &'a [i64],
    pub(super) images_seq_mask: &'a [bool],
    pub(super) image_array: &'a [f32],
}

pub(super) struct DecodeFeedInputs {
    pub(super) input_id: i64,
    pub(super) position_id: i64,
    pub(super) cache: KvCache,
}

struct PadRequest<'a, T> {
    values: &'a [T],
    fixed_length: Option<usize>,
    pad_value: T,
    input_name: &'a str,
}

pub(super) fn inspect_input_metadata(session: &Session) -> anyhow::Result<InputMetadata> {
    let mut names = HashSet::new();
    let mut image_dtype = None;
    let mut fixed_sequence_length = None;
    let mut fixed_image_size = None;
    let mut past_inputs = Vec::new();

    for input in session.inputs() {
        names.insert(input.name().to_string());
        if is_past_cache_name(input.name()) {
            past_inputs.push(input.name().to_string());
        }

        let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
            continue;
        };

        if input.name() == "images_ori" {
            image_dtype = Some(*ty);
            fixed_image_size = fixed_axis(shape, 2)
                .map(u32::try_from)
                .transpose()
                .map_err(|_| anyhow!("images_ori fixed image size is invalid"))?;
        }

        if input.name() == "input_ids" {
            fixed_sequence_length = fixed_axis(shape, 1)
                .map(usize::try_from)
                .transpose()
                .map_err(|_| anyhow!("input_ids fixed sequence length is invalid"))?;
        }
    }

    let present_outputs = session
        .outputs()
        .iter()
        .filter(|output| is_present_cache_name(output.name()))
        .map(|output| output.name().to_string())
        .collect();

    let image_dtype = image_dtype.ok_or_else(|| {
        anyhow!(
            "ONNX graph does not expose required input `images_ori`; found inputs: {:?}",
            names
        )
    })?;

    Ok(InputMetadata {
        names,
        image_dtype,
        fixed_sequence_length,
        kv_cache: KvCacheMetadata::new(past_inputs, present_outputs),
        fixed_image_size,
    })
}

pub(super) fn inspect_decode_input_metadata(
    session: &Session,
) -> anyhow::Result<DecodeInputMetadata> {
    let names = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect::<HashSet<_>>();
    let past_inputs = session
        .inputs()
        .iter()
        .filter(|input| is_past_cache_name(input.name()))
        .map(|input| input.name().to_string())
        .collect();
    let present_outputs = session
        .outputs()
        .iter()
        .filter(|output| is_present_cache_name(output.name()))
        .map(|output| output.name().to_string())
        .collect();

    Ok(DecodeInputMetadata {
        names,
        kv_cache: KvCacheMetadata::new(past_inputs, present_outputs),
    })
}

pub(super) fn validate_image_size(metadata: &InputMetadata, configured: u32) -> anyhow::Result<()> {
    if let Some(expected) = metadata.fixed_image_size
        && expected != configured
    {
        return Err(anyhow!(
            "ONNX graph expects image_size={expected}, but config uses image_size={configured}"
        ));
    }

    Ok(())
}

pub(super) fn make_feeds(
    metadata: &InputMetadata,
    image_size: u32,
    inputs: FeedInputs<'_>,
) -> anyhow::Result<FeedValues> {
    let mut feeds = Vec::new();
    append_input_ids(&mut feeds, metadata, inputs.input_ids)?;
    append_attention_mask(&mut feeds, metadata, inputs.input_ids)?;
    append_image(&mut feeds, metadata, image_size, inputs.image_array)?;
    append_image_crop(&mut feeds, metadata, image_size)?;
    append_image_sequence_mask(&mut feeds, metadata, inputs.images_seq_mask)?;
    append_spatial_crop(&mut feeds, metadata)?;
    Ok(feeds)
}

pub(super) fn make_decode_feeds(
    metadata: &DecodeInputMetadata,
    inputs: DecodeFeedInputs,
) -> anyhow::Result<FeedValues> {
    let mut feeds = Vec::new();
    if metadata.names.contains("input_ids") {
        feeds.push((
            "input_ids".into(),
            Tensor::<i64>::from_array((sequence_shape(1), vec![inputs.input_id]))
                .map_err(|err| anyhow!("failed to create decode input_ids tensor: {err}"))?
                .into(),
        ));
    }
    if metadata.names.contains("position_ids") {
        feeds.push((
            "position_ids".into(),
            Tensor::<i64>::from_array((sequence_shape(1), vec![inputs.position_id]))
                .map_err(|err| anyhow!("failed to create position_ids tensor: {err}"))?
                .into(),
        ));
    }

    let past_inputs = metadata.kv_cache.past_inputs();
    if past_inputs.len() != inputs.cache.values.len() {
        return Err(anyhow!(
            "decode graph expects {} cache tensors, but runtime has {}",
            past_inputs.len(),
            inputs.cache.values.len()
        ));
    }

    for (name, value) in past_inputs.iter().zip(inputs.cache.values) {
        feeds.push((name.clone().into(), value.into()));
    }

    Ok(feeds)
}

pub(super) fn collect_present_cache(
    outputs: &mut SessionOutputs<'_>,
    metadata: &KvCacheMetadata,
) -> anyhow::Result<KvCache> {
    collect_present_cache_with_limit(outputs, metadata, None)
}

pub(super) fn collect_present_cache_trimmed(
    outputs: &mut SessionOutputs<'_>,
    metadata: &KvCacheMetadata,
    sequence_len: usize,
) -> anyhow::Result<KvCache> {
    collect_present_cache_with_limit(outputs, metadata, Some(sequence_len))
}

fn collect_present_cache_with_limit(
    outputs: &mut SessionOutputs<'_>,
    metadata: &KvCacheMetadata,
    sequence_len: Option<usize>,
) -> anyhow::Result<KvCache> {
    let mut values = Vec::with_capacity(metadata.present_outputs.len());
    for name in &metadata.present_outputs {
        let value = outputs
            .remove(name)
            .ok_or_else(|| anyhow!("ONNX output `{name}` is missing"))?;
        values.push(match sequence_len {
            Some(sequence_len) => trim_cache_tensor(value, name, sequence_len)?,
            None => value,
        });
    }

    Ok(KvCache { values })
}

fn trim_cache_tensor(value: DynValue, name: &str, sequence_len: usize) -> anyhow::Result<DynValue> {
    match value.dtype() {
        ValueType::Tensor {
            ty: TensorElementType::Float32,
            ..
        } => trim_typed_cache_tensor::<f32>(&value, name, sequence_len),
        ValueType::Tensor {
            ty: TensorElementType::Float16,
            ..
        } => trim_typed_cache_tensor::<f16>(&value, name, sequence_len),
        ValueType::Tensor {
            ty: TensorElementType::Bfloat16,
            ..
        } => trim_typed_cache_tensor::<bf16>(&value, name, sequence_len),
        other => Err(anyhow!(
            "cache tensor `{name}` has unsupported dtype: {other:?}"
        )),
    }
}

fn trim_typed_cache_tensor<
    T: Copy
        + ort::value::IntoTensorElementType
        + ort::value::PrimitiveTensorElementType
        + std::fmt::Debug
        + 'static,
>(
    value: &DynValue,
    name: &str,
    sequence_len: usize,
) -> anyhow::Result<DynValue> {
    let (shape, data) = value
        .try_extract_tensor::<T>()
        .map_err(|err| anyhow!("cache tensor `{name}` extraction failed: {err}"))?;
    let shape_values = shape.iter().copied().collect::<Vec<_>>();
    if shape_values.len() != 4 {
        return Err(anyhow!(
            "cache tensor `{name}` has invalid shape {shape_values:?}"
        ));
    }

    let batch = usize::try_from(shape_values[0])
        .map_err(|_| anyhow!("cache tensor `{name}` batch dimension is invalid"))?;
    let heads = usize::try_from(shape_values[1])
        .map_err(|_| anyhow!("cache tensor `{name}` heads dimension is invalid"))?;
    let total_sequence = usize::try_from(shape_values[2])
        .map_err(|_| anyhow!("cache tensor `{name}` sequence dimension is invalid"))?;
    let head_dim = usize::try_from(shape_values[3])
        .map_err(|_| anyhow!("cache tensor `{name}` head dimension is invalid"))?;

    if sequence_len > total_sequence {
        return Err(anyhow!(
            "cache tensor `{name}` has sequence length {total_sequence}, cannot trim to {sequence_len}"
        ));
    }

    let stride = total_sequence
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?;
    let trimmed_stride = sequence_len
        .checked_mul(head_dim)
        .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?;
    let mut trimmed = Vec::with_capacity(
        batch
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(trimmed_stride))
            .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?,
    );

    for batch_idx in 0..batch {
        for head_idx in 0..heads {
            let start = (batch_idx
                .checked_mul(heads)
                .and_then(|value| value.checked_add(head_idx))
                .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?)
            .checked_mul(stride)
            .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?;
            let end = start
                .checked_add(trimmed_stride)
                .ok_or_else(|| anyhow!("cache tensor `{name}` shape is too large"))?;
            trimmed.extend_from_slice(data.get(start..end).ok_or_else(|| {
                anyhow!("cache tensor `{name}` data length does not match shape {shape_values:?}")
            })?);
        }
    }

    let trimmed_shape = Shape::from([
        shape_values[0],
        shape_values[1],
        sequence_len as i64,
        shape_values[3],
    ]);
    Ok(Tensor::<T>::from_array((trimmed_shape, trimmed))
        .map_err(|err| anyhow!("failed to create trimmed cache tensor `{name}`: {err}"))?
        .into_dyn())
}

fn append_input_ids(
    feeds: &mut FeedValues,
    metadata: &InputMetadata,
    input_ids: &[i64],
) -> anyhow::Result<()> {
    if metadata.names.contains("input_ids") {
        let values = prepare_1d(PadRequest {
            values: input_ids,
            fixed_length: metadata.fixed_sequence_length,
            pad_value: 0,
            input_name: "input_ids",
        })?;
        feeds.push((
            "input_ids".into(),
            Tensor::<i64>::from_array((sequence_shape(values.len()), values))
                .map_err(|err| anyhow!("failed to create input_ids tensor: {err}"))?
                .into(),
        ));
    }

    Ok(())
}

fn append_attention_mask(
    feeds: &mut FeedValues,
    metadata: &InputMetadata,
    input_ids: &[i64],
) -> anyhow::Result<()> {
    if metadata.names.contains("attention_mask") {
        let attention_mask = vec![1_i64; input_ids.len()];
        let values = prepare_1d(PadRequest {
            values: &attention_mask,
            fixed_length: metadata.fixed_sequence_length,
            pad_value: 0,
            input_name: "attention_mask",
        })?;
        feeds.push((
            "attention_mask".into(),
            Tensor::<i64>::from_array((sequence_shape(values.len()), values))
                .map_err(|err| anyhow!("failed to create attention_mask tensor: {err}"))?
                .into(),
        ));
    }

    Ok(())
}

fn append_image(
    feeds: &mut FeedValues,
    metadata: &InputMetadata,
    image_size: u32,
    image_array: &[f32],
) -> anyhow::Result<()> {
    if metadata.names.contains("images_ori") {
        feeds.push((
            "images_ori".into(),
            image_tensor(image_array, image_size, metadata.image_dtype)?,
        ));
    }

    Ok(())
}

fn append_image_crop(
    feeds: &mut FeedValues,
    metadata: &InputMetadata,
    image_size: u32,
) -> anyhow::Result<()> {
    if metadata.names.contains("images_crop") {
        let values = vec![0.0_f32; 3 * image_size as usize * image_size as usize];
        feeds.push((
            "images_crop".into(),
            image_tensor(&values, image_size, metadata.image_dtype)?,
        ));
    }

    Ok(())
}

fn append_image_sequence_mask(
    feeds: &mut FeedValues,
    metadata: &InputMetadata,
    images_seq_mask: &[bool],
) -> anyhow::Result<()> {
    if metadata.names.contains("images_seq_mask") {
        let values = prepare_1d(PadRequest {
            values: images_seq_mask,
            fixed_length: metadata.fixed_sequence_length,
            pad_value: false,
            input_name: "images_seq_mask",
        })?;
        feeds.push((
            "images_seq_mask".into(),
            Tensor::<bool>::from_array((sequence_shape(values.len()), values))
                .map_err(|err| anyhow!("failed to create images_seq_mask tensor: {err}"))?
                .into(),
        ));
    }

    Ok(())
}

fn append_spatial_crop(feeds: &mut FeedValues, metadata: &InputMetadata) -> anyhow::Result<()> {
    if metadata.names.contains("images_spatial_crop") {
        feeds.push((
            "images_spatial_crop".into(),
            Tensor::<i64>::from_array((Shape::from([1_i64, 2]), vec![1_i64, 1]))
                .map_err(|err| anyhow!("failed to create images_spatial_crop tensor: {err}"))?
                .into(),
        ));
    }

    Ok(())
}

fn fixed_axis(shape: &Shape, axis: usize) -> Option<i64> {
    shape.get(axis).copied().filter(|dimension| *dimension > 0)
}

fn is_past_cache_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.starts_with("past_key_values.")
        || normalized.starts_with("past_key_values_")
        || normalized.starts_with("past.")
        || normalized.starts_with("past_")
}

fn is_present_cache_name(name: &str) -> bool {
    let normalized = name.to_ascii_lowercase();
    normalized.starts_with("present.")
        || normalized.starts_with("present_")
        || normalized.starts_with("present_key_values.")
        || normalized.starts_with("present_key_values_")
}

fn prepare_1d<T: Copy>(request: PadRequest<'_, T>) -> anyhow::Result<Vec<T>> {
    let Some(fixed_length) = request.fixed_length else {
        return Ok(request.values.to_vec());
    };
    if request.values.len() > fixed_length {
        return Err(anyhow!(
            "input `{}` has {} tokens, but the ONNX graph is fixed to {fixed_length}",
            request.input_name,
            request.values.len()
        ));
    }

    let mut output = vec![request.pad_value; fixed_length];
    output[..request.values.len()].copy_from_slice(request.values);
    Ok(output)
}

fn sequence_shape(values_len: usize) -> Shape {
    Shape::from([1_i64, values_len as i64])
}

fn image_tensor(
    values: &[f32],
    image_size: u32,
    dtype: TensorElementType,
) -> anyhow::Result<SessionInputValue<'static>> {
    let shape = Shape::from([1_i64, 3, i64::from(image_size), i64::from(image_size)]);
    match dtype {
        TensorElementType::Float32 => Ok(Tensor::<f32>::from_array((shape, values.to_vec()))
            .map_err(|err| anyhow!("failed to create f32 image tensor: {err}"))?
            .into()),
        TensorElementType::Float16 => Ok(Tensor::<f16>::from_array((
            shape,
            values
                .iter()
                .copied()
                .map(f16::from_f32)
                .collect::<Vec<_>>(),
        ))
        .map_err(|err| anyhow!("failed to create f16 image tensor: {err}"))?
        .into()),
        TensorElementType::Bfloat16 => Ok(Tensor::<bf16>::from_array((
            shape,
            values
                .iter()
                .copied()
                .map(bf16::from_f32)
                .collect::<Vec<_>>(),
        ))
        .map_err(|err| anyhow!("failed to create bf16 image tensor: {err}"))?
        .into()),
        other => Err(anyhow!(
            "unsupported images_ori input dtype `{other}`; expected f32, f16, or bf16"
        )),
    }
}

#[cfg(test)]
pub(super) fn prepare_i64_for_test(
    values: &[i64],
    fixed_length: Option<usize>,
) -> anyhow::Result<Vec<i64>> {
    prepare_1d(PadRequest {
        values,
        fixed_length,
        pad_value: 0,
        input_name: "input_ids",
    })
}

impl KvCacheMetadata {
    fn new(mut past_inputs: Vec<String>, mut present_outputs: Vec<String>) -> Self {
        past_inputs.sort();
        present_outputs.sort();

        Self {
            past_inputs,
            present_outputs,
        }
    }

    pub(super) fn is_supported(&self) -> bool {
        !self.past_inputs.is_empty() && self.past_inputs.len() == self.present_outputs.len()
    }

    pub(super) fn can_seed_decode_cache(&self, decode: &Self) -> bool {
        !self.present_outputs.is_empty() && self.present_outputs.len() == decode.past_inputs.len()
    }

    pub(super) fn has_present_outputs(&self) -> bool {
        !self.present_outputs.is_empty()
    }

    pub(super) fn past_inputs(&self) -> &[String] {
        &self.past_inputs
    }

    pub(super) fn summary(&self) -> String {
        format!(
            "past_inputs={} present_outputs={} supported={}",
            self.past_inputs.len(),
            self.present_outputs.len(),
            self.is_supported()
        )
    }
}

#[cfg(test)]
pub(super) fn kv_cache_metadata_for_test(
    past_inputs: Vec<String>,
    present_outputs: Vec<String>,
) -> KvCacheMetadata {
    KvCacheMetadata::new(past_inputs, present_outputs)
}

#[cfg(test)]
pub(super) fn prepare_bool_for_test(
    values: &[bool],
    fixed_length: Option<usize>,
) -> anyhow::Result<Vec<bool>> {
    prepare_1d(PadRequest {
        values,
        fixed_length,
        pad_value: false,
        input_name: "images_seq_mask",
    })
}

#[cfg(test)]
mod tests {
    use ort::value::{DynValue, Tensor};

    use super::trim_cache_tensor;

    #[test]
    fn trims_cache_tensor_sequence_axis() {
        let value: DynValue = Tensor::<f32>::from_array((
            [1_usize, 2, 4, 2],
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
        ))
        .unwrap()
        .into_dyn();

        let trimmed = trim_cache_tensor(value, "present.0.key", 2).unwrap();
        let (shape, data) = trimmed.try_extract_tensor::<f32>().unwrap();

        assert_eq!(shape.iter().copied().collect::<Vec<_>>(), vec![1, 2, 2, 2]);
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0]);
    }
}
