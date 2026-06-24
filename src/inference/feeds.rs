use std::{borrow::Cow, collections::HashSet};

use anyhow::anyhow;
use half::{bf16, f16};
use ort::{
    session::{Session, SessionInputValue},
    value::{Shape, Tensor, TensorElementType, ValueType},
};

type FeedValues = Vec<(Cow<'static, str>, SessionInputValue<'static>)>;

#[derive(Debug, Clone)]
pub(super) struct InputMetadata {
    pub(super) names: HashSet<String>,
    pub(super) image_dtype: TensorElementType,
    pub(super) fixed_sequence_length: Option<usize>,
    fixed_image_size: Option<u32>,
}

pub(super) struct FeedInputs<'a> {
    pub(super) input_ids: &'a [i64],
    pub(super) images_seq_mask: &'a [bool],
    pub(super) image_array: &'a [f32],
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

    for input in session.inputs() {
        names.insert(input.name().to_string());
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
        fixed_image_size,
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
