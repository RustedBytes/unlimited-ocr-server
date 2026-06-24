mod model;
mod tensor;

use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use anyhow::{Context, anyhow};
use half::{bf16, f16};
use image::{
    DynamicImage, GenericImage, GenericImageView, ImageDecoder, ImageReader, Rgb, RgbImage,
    imageops::FilterType,
};
use log::{debug, info, trace, warn};
use ort::{
    environment::Environment,
    session::{Session, SessionInputValue},
    value::{Shape, Tensor, TensorElementType, ValueType},
};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::{
    config::{Config, ModelVariant},
    types::{GenerationMetadata, InferenceMetadata, TaskSpec, TensorMetadata},
};

use self::{
    model::{load_session, tokenizer_path_for_model},
    tensor::{TensorData, argmax_token_at_position, extract_output_tensor, tensor_metadata_f32},
};

const DEFAULT_PROMPT: &str = "<image>Free OCR.";
const IMAGE_TOKEN: &str = "<image>";
const TASK_TOKEN: &str = "unlimited_ocr";
const BOS_TOKEN_ID: i64 = 0;
const EOS_TOKEN_ID: i64 = 1;
const IMAGE_TOKEN_ID: i64 = 128_815;
const PATCH_SIZE: u32 = 16;
const DOWNSAMPLE_RATIO: u32 = 4;
const GRAY_PAD: Rgb<u8> = Rgb([127, 127, 127]);

pub fn validate_model_artifacts(model_path: &Path, _variant: ModelVariant) -> anyhow::Result<()> {
    if !model_path.exists() {
        return Err(anyhow!(
            "model file does not exist: {}",
            model_path.display()
        ));
    }

    let tokenizer_path = tokenizer_path_for_model(model_path)?;
    if !tokenizer_path.exists() {
        return Err(anyhow!(
            "tokenizer file does not exist: {}",
            tokenizer_path.display()
        ));
    }

    Ok(())
}

pub struct UnlimitedOcrWorker {
    id: usize,
    model_path: PathBuf,
    model_variant: ModelVariant,
    session: Option<Session>,
    tokenizer: Tokenizer,
    backend: String,
    input_metadata: InputMetadata,
    max_new_tokens: usize,
    image_size: u32,
    execution_providers: Vec<String>,
}

#[derive(Debug, Clone)]
struct InputMetadata {
    names: HashSet<String>,
    image_dtype: TensorElementType,
    fixed_sequence_length: Option<usize>,
    fixed_image_size: Option<u32>,
}

#[derive(Debug)]
struct PromptInputs {
    input_ids: Vec<i64>,
    images_seq_mask: Vec<bool>,
}

impl UnlimitedOcrWorker {
    pub fn new(id: usize, config: Arc<Config>) -> anyhow::Result<Self> {
        let started = Instant::now();
        validate_model_artifacts(&config.model_path, config.model_variant)?;

        info!(
            "initializing Unlimited-OCR worker worker_id={} model={}",
            id,
            config.model_path.display()
        );

        let env =
            Environment::current().context("failed to initialize ONNX Runtime environment")?;
        let devices = env
            .devices()
            .map(|device| {
                let ep = device.ep().unwrap_or("unknown").to_string();
                let ty = format!("{:?}", device.ty());
                format!("{ep}:{ty}")
            })
            .collect::<Vec<_>>();

        if devices.is_empty() {
            warn!("ONNX Runtime did not report accelerator devices; CPU fallback will be used");
        } else {
            info!(
                "ONNX Runtime detected devices worker_id={} devices={:?}",
                id, devices
            );
        }

        let session_started = Instant::now();
        let session = load_session(&config.model_path, &config.execution_providers)?;
        let input_metadata = inspect_input_metadata(&session)?;
        validate_image_size(&input_metadata, config.model_image_size)?;
        info!(
            "ONNX session loaded worker_id={} elapsed_ms={}",
            id,
            session_started.elapsed().as_millis()
        );

        let tokenizer_path = tokenizer_path_for_model(&config.model_path)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| {
            anyhow!(
                "failed to load tokenizer from {}: {err}",
                tokenizer_path.display()
            )
        })?;

        let worker = Self {
            id,
            model_path: config.model_path.clone(),
            model_variant: config.model_variant,
            session: Some(session),
            tokenizer,
            backend: format!("ort:auto:max_performance:{}", devices.join(",")),
            input_metadata,
            max_new_tokens: config.max_new_tokens,
            image_size: config.model_image_size,
            execution_providers: config.execution_providers.clone(),
        };

        info!(
            "Unlimited-OCR worker initialized worker_id={} backend={} image_size={} max_new_tokens={} execution_providers={:?} elapsed_ms={}",
            id,
            worker.backend,
            worker.image_size,
            worker.max_new_tokens,
            worker.execution_providers,
            started.elapsed().as_millis()
        );

        Ok(worker)
    }

    pub fn take_for_blocking(&mut self) -> Self {
        // ONNX sessions are consumed by a blocking thread for inference and
        // returned afterward. Option::take prevents two threads from touching
        // the same session handle at once.
        Self {
            id: self.id,
            model_path: self.model_path.clone(),
            model_variant: self.model_variant,
            session: self.session.take(),
            tokenizer: self.tokenizer.clone(),
            backend: self.backend.clone(),
            input_metadata: self.input_metadata.clone(),
            max_new_tokens: self.max_new_tokens,
            image_size: self.image_size,
            execution_providers: self.execution_providers.clone(),
        }
    }

    pub fn infer(
        &mut self,
        image_path: &Path,
        task: &TaskSpec,
    ) -> anyhow::Result<InferenceMetadata> {
        let total_started = Instant::now();
        let prompt = prompt_from_task(task);
        info!(
            "Unlimited-OCR inference started worker_id={} prompt_present={} image_path={}",
            self.id,
            task.text_input.is_some(),
            image_path.display()
        );

        let decode_started = Instant::now();
        let image = decode_image_with_orientation(image_path)?;
        let (original_width, original_height) = image.dimensions();
        debug!(
            "image decoded worker_id={} image_path={} width={} height={} elapsed_ms={}",
            self.id,
            image_path.display(),
            original_width,
            original_height,
            decode_started.elapsed().as_millis()
        );

        let preprocess_started = Instant::now();
        let image_array = preprocess_image(image, self.image_size)?;
        debug!(
            "image preprocessed worker_id={} image_size={} output_f32_values={} elapsed_ms={}",
            self.id,
            self.image_size,
            image_array.len(),
            preprocess_started.elapsed().as_millis()
        );

        let PromptInputs {
            mut input_ids,
            mut images_seq_mask,
        } = build_image_prompt(&self.tokenizer, &prompt, self.image_size)?;
        let generated_ids = self.generate(&mut input_ids, &mut images_seq_mask, &image_array)?;
        let generated_text = decode_generated_text(&self.tokenizer, &generated_ids)?;
        let result = json!({ "text": generated_text });
        let generation = GenerationMetadata {
            task_token: TASK_TOKEN.to_string(),
            prompt_text: prompt,
            generated_text: generated_text.clone(),
            generated_tokens: generated_ids.len(),
            result: result.clone(),
        };
        let elapsed_ms = total_started.elapsed().as_millis();

        let outputs = vec![
            tensor_metadata_f32(
                "images_ori",
                &Shape::from([
                    1_i64,
                    3,
                    i64::from(self.image_size),
                    i64::from(self.image_size),
                ]),
                &image_array,
            ),
            TensorMetadata {
                name: "generated_token_ids".to_string(),
                shape: vec![1, generated_ids.len() as i64],
                elements: generated_ids.len(),
                mean: None,
                min: None,
                max: None,
            },
        ];

        info!(
            "Unlimited-OCR generation finished worker_id={} model_variant={} generated_tokens={} elapsed_ms={}",
            self.id,
            self.model_variant.as_str(),
            generated_ids.len(),
            elapsed_ms
        );

        Ok(InferenceMetadata {
            backend: self.backend.clone(),
            model_path: self.model_path.clone(),
            model_variant: self.model_variant,
            input_name: "images_ori".to_string(),
            input_dtype: self.input_metadata.image_dtype.to_string(),
            original_width,
            original_height,
            processed_width: self.image_size,
            processed_height: self.image_size,
            elapsed_ms,
            task_token: generation.task_token.clone(),
            prompt_text: generation.prompt_text.clone(),
            generated_text,
            generated_tokens: generation.generated_tokens,
            result,
            generations: vec![generation],
            outputs,
        })
    }

    fn generate(
        &mut self,
        input_ids: &mut Vec<i64>,
        images_seq_mask: &mut Vec<bool>,
        image_array: &[f32],
    ) -> anyhow::Result<Vec<i64>> {
        let mut generated = Vec::new();

        for _ in 0..self.max_new_tokens {
            let position = input_ids
                .len()
                .checked_sub(1)
                .ok_or_else(|| anyhow!("input_ids cannot be empty"))?;
            let logits = self.run_model(input_ids, images_seq_mask, image_array)?;
            let next_token_id = argmax_token_at_position(&logits, position)?;

            input_ids.push(next_token_id);
            images_seq_mask.push(false);
            generated.push(next_token_id);

            if next_token_id == EOS_TOKEN_ID {
                break;
            }
        }

        Ok(generated)
    }

    fn run_model(
        &mut self,
        input_ids: &[i64],
        images_seq_mask: &[bool],
        image_array: &[f32],
    ) -> anyhow::Result<TensorData> {
        let feeds = self
            .make_feeds(input_ids, images_seq_mask, image_array)
            .map_err(|err| {
                if err.to_string().contains("fixed to") {
                    anyhow!(
                        "{err}. Re-export with --dynamic-image or increase --image-sequence-length."
                    )
                } else {
                    err
                }
            })?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("worker {} ONNX session is unavailable", self.id))?;
        let outputs = session
            .run(feeds)
            .map_err(|err| anyhow!("Unlimited-OCR ONNX inference failed: {err}"))?;

        extract_output_tensor(&outputs[0], "logits")
    }

    fn make_feeds(
        &self,
        input_ids: &[i64],
        images_seq_mask: &[bool],
        image_array: &[f32],
    ) -> anyhow::Result<Vec<(std::borrow::Cow<'static, str>, SessionInputValue<'static>)>> {
        let mut feeds = Vec::new();

        if self.input_metadata.names.contains("input_ids") {
            let values = prepare_i64_1d(
                input_ids,
                self.input_metadata.fixed_sequence_length,
                0,
                "input_ids",
            )?;
            feeds.push((
                "input_ids".into(),
                Tensor::<i64>::from_array((sequence_shape(values.len()), values))
                    .map_err(|err| anyhow!("failed to create input_ids tensor: {err}"))?
                    .into(),
            ));
        }

        if self.input_metadata.names.contains("attention_mask") {
            let attention_mask = vec![1_i64; input_ids.len()];
            let values = prepare_i64_1d(
                &attention_mask,
                self.input_metadata.fixed_sequence_length,
                0,
                "attention_mask",
            )?;
            feeds.push((
                "attention_mask".into(),
                Tensor::<i64>::from_array((sequence_shape(values.len()), values))
                    .map_err(|err| anyhow!("failed to create attention_mask tensor: {err}"))?
                    .into(),
            ));
        }

        if self.input_metadata.names.contains("images_ori") {
            feeds.push((
                "images_ori".into(),
                image_tensor(
                    image_array,
                    self.image_size,
                    self.input_metadata.image_dtype,
                )?,
            ));
        }

        if self.input_metadata.names.contains("images_crop") {
            let values = vec![0.0_f32; 3 * self.image_size as usize * self.image_size as usize];
            feeds.push((
                "images_crop".into(),
                image_tensor(&values, self.image_size, self.input_metadata.image_dtype)?,
            ));
        }

        if self.input_metadata.names.contains("images_seq_mask") {
            let values = prepare_bool_1d(
                images_seq_mask,
                self.input_metadata.fixed_sequence_length,
                false,
                "images_seq_mask",
            )?;
            feeds.push((
                "images_seq_mask".into(),
                Tensor::<bool>::from_array((sequence_shape(values.len()), values))
                    .map_err(|err| anyhow!("failed to create images_seq_mask tensor: {err}"))?
                    .into(),
            ));
        }

        if self.input_metadata.names.contains("images_spatial_crop") {
            feeds.push((
                "images_spatial_crop".into(),
                Tensor::<i64>::from_array((Shape::from([1_i64, 2]), vec![1_i64, 1]))
                    .map_err(|err| anyhow!("failed to create images_spatial_crop tensor: {err}"))?
                    .into(),
            ));
        }

        Ok(feeds)
    }
}

fn inspect_input_metadata(session: &Session) -> anyhow::Result<InputMetadata> {
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

fn fixed_axis(shape: &Shape, axis: usize) -> Option<i64> {
    shape.get(axis).copied().filter(|dimension| *dimension > 0)
}

fn validate_image_size(metadata: &InputMetadata, configured: u32) -> anyhow::Result<()> {
    if let Some(expected) = metadata.fixed_image_size
        && expected != configured
    {
        return Err(anyhow!(
            "ONNX graph expects image_size={expected}, but config uses image_size={configured}"
        ));
    }

    Ok(())
}

fn prompt_from_task(task: &TaskSpec) -> String {
    task.text_input
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_PROMPT)
        .to_string()
}

fn build_image_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    image_size: u32,
) -> anyhow::Result<PromptInputs> {
    let prompt = if prompt.contains(IMAGE_TOKEN) {
        prompt.to_string()
    } else {
        format!("{IMAGE_TOKEN}{prompt}")
    };
    let (before, after) = prompt
        .split_once(IMAGE_TOKEN)
        .ok_or_else(|| anyhow!("prompt must contain `{IMAGE_TOKEN}`"))?;
    let before_ids = encode_text(tokenizer, before)?;
    let after_ids = encode_text(tokenizer, after)?;
    let image_ids = vec![IMAGE_TOKEN_ID; image_token_count(image_size)];

    let mut input_ids =
        Vec::with_capacity(1 + before_ids.len() + image_ids.len() + after_ids.len());
    input_ids.push(BOS_TOKEN_ID);
    input_ids.extend(before_ids.iter().copied());
    input_ids.extend(image_ids.iter().copied());
    input_ids.extend(after_ids.iter().copied());

    let mut images_seq_mask = Vec::with_capacity(input_ids.len());
    images_seq_mask.extend(std::iter::repeat_n(false, 1 + before_ids.len()));
    images_seq_mask.extend(std::iter::repeat_n(true, image_ids.len()));
    images_seq_mask.extend(std::iter::repeat_n(false, after_ids.len()));

    Ok(PromptInputs {
        input_ids,
        images_seq_mask,
    })
}

fn encode_text(tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|err| anyhow!("failed to tokenize prompt fragment `{text}`: {err}"))?;
    Ok(encoding.get_ids().iter().map(|id| i64::from(*id)).collect())
}

fn image_token_count(image_size: u32) -> usize {
    let image_patches = image_size / PATCH_SIZE;
    let num_queries = image_patches.div_ceil(DOWNSAMPLE_RATIO);
    ((num_queries + 1) * num_queries + 1) as usize
}

fn prepare_i64_1d(
    values: &[i64],
    fixed_length: Option<usize>,
    pad_value: i64,
    input_name: &str,
) -> anyhow::Result<Vec<i64>> {
    prepare_1d(values, fixed_length, pad_value, input_name)
}

fn prepare_bool_1d(
    values: &[bool],
    fixed_length: Option<usize>,
    pad_value: bool,
    input_name: &str,
) -> anyhow::Result<Vec<bool>> {
    prepare_1d(values, fixed_length, pad_value, input_name)
}

fn prepare_1d<T: Copy>(
    values: &[T],
    fixed_length: Option<usize>,
    pad_value: T,
    input_name: &str,
) -> anyhow::Result<Vec<T>> {
    let Some(fixed_length) = fixed_length else {
        return Ok(values.to_vec());
    };
    if values.len() > fixed_length {
        return Err(anyhow!(
            "input `{input_name}` has {} tokens, but the ONNX graph is fixed to {fixed_length}",
            values.len()
        ));
    }

    let mut output = vec![pad_value; fixed_length];
    output[..values.len()].copy_from_slice(values);
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

fn decode_image_with_orientation(image_path: &Path) -> anyhow::Result<DynamicImage> {
    let reader = ImageReader::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?
        .with_guessed_format()
        .context("failed to guess image format")?;
    let mut decoder = reader
        .into_decoder()
        .context("failed to create image decoder")?;
    let orientation = decoder
        .orientation()
        .context("failed to read image orientation")?;
    let mut image = DynamicImage::from_decoder(decoder).context("failed to decode image")?;
    image.apply_orientation(orientation);
    Ok(image)
}

fn preprocess_image(image: DynamicImage, image_size: u32) -> anyhow::Result<Vec<f32>> {
    if image_size == 0 {
        return Err(anyhow!("image_size must be greater than zero"));
    }

    trace!(
        "normalizing Unlimited-OCR image target_width={} target_height={}",
        image_size, image_size
    );

    let image = image.to_rgb8();
    let contained = if image_size <= 640 {
        DynamicImage::ImageRgb8(image)
            .resize_exact(image_size, image_size, FilterType::CatmullRom)
            .to_rgb8()
    } else {
        resize_to_fit(&image, image_size)?
    };
    let padded = pad_to_square(&contained, image_size)?;
    Ok(normalize_chw(&padded))
}

fn resize_to_fit(image: &RgbImage, image_size: u32) -> anyhow::Result<RgbImage> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(anyhow!("image dimensions must be greater than zero"));
    }

    let scale =
        (f64::from(image_size) / f64::from(width)).min(f64::from(image_size) / f64::from(height));
    let resized_width = ((f64::from(width) * scale).round() as u32).max(1);
    let resized_height = ((f64::from(height) * scale).round() as u32).max(1);

    Ok(DynamicImage::ImageRgb8(image.clone())
        .resize_exact(resized_width, resized_height, FilterType::CatmullRom)
        .to_rgb8())
}

fn pad_to_square(image: &RgbImage, image_size: u32) -> anyhow::Result<RgbImage> {
    let (width, height) = image.dimensions();
    if width > image_size || height > image_size {
        return Err(anyhow!(
            "resized image {width}x{height} does not fit target square {image_size}x{image_size}"
        ));
    }

    let mut output = RgbImage::from_pixel(image_size, image_size, GRAY_PAD);
    let x = (image_size - width) / 2;
    let y = (image_size - height) / 2;
    output
        .copy_from(image, x, y)
        .map_err(|err| anyhow!("failed to pad image: {err}"))?;
    Ok(output)
}

fn normalize_chw(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let plane = (width * height) as usize;
    let mut chw = vec![0.0_f32; 3 * plane];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            let dst = y * width as usize + x;
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                chw[channel * plane + dst] = (value - 0.5) / 0.5;
            }
        }
    }

    chw
}

fn decode_generated_text(tokenizer: &Tokenizer, generated_ids: &[i64]) -> anyhow::Result<String> {
    let generated_u32 = generated_ids
        .iter()
        .filter_map(|id| u32::try_from(*id).ok())
        .collect::<Vec<_>>();
    let raw_text = tokenizer
        .decode(&generated_u32, false)
        .map_err(|err| anyhow!("failed to decode generated tokens: {err}"))?;
    Ok(clean_generated_text(&raw_text))
}

fn clean_generated_text(raw: &str) -> String {
    raw.strip_suffix("<｜end▁of▁sentence｜>")
        .unwrap_or(raw)
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests;
