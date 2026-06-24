mod feeds;
mod image;
mod model;
mod prompt;
mod tensor;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

use ::image::GenericImageView;
use anyhow::{Context, anyhow};
use log::{debug, info, warn};
use ort::{environment::Environment, session::Session, value::Shape};
use serde_json::json;
use tokenizers::Tokenizer;

use crate::{
    config::{Config, ModelVariant},
    types::{GenerationMetadata, InferenceMetadata, TaskSpec, TensorMetadata},
};

use self::{
    feeds::{FeedInputs, InputMetadata, inspect_input_metadata, make_feeds, validate_image_size},
    image::{decode_image_with_orientation, preprocess_image},
    model::{load_session, tokenizer_path_for_model},
    prompt::{
        EOS_TOKEN_ID, PromptInputs, build_image_prompt, decode_generated_text, prompt_from_task,
    },
    tensor::{TensorData, argmax_token_at_position, extract_output_tensor, tensor_metadata_f32},
};

const TASK_TOKEN: &str = "unlimited_ocr";

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

struct GenerationState<'a> {
    input_ids: Vec<i64>,
    images_seq_mask: Vec<bool>,
    image_array: &'a [f32],
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

        let devices = detected_devices(id)?;
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

        let (image_array, original_width, original_height) = self.prepare_image(image_path)?;
        let prompt_inputs = build_image_prompt(&self.tokenizer, &prompt, self.image_size)?;
        let generated_ids = self.generate(GenerationState::new(prompt_inputs, &image_array))?;
        let generated_text = decode_generated_text(&self.tokenizer, &generated_ids)?;
        let generation = generation_metadata(prompt, generated_text.clone(), generated_ids.len());
        let result = generation.result.clone();
        let elapsed_ms = total_started.elapsed().as_millis();

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
            outputs: self.output_metadata(&image_array, generated_ids.len()),
        })
    }

    fn prepare_image(&self, image_path: &Path) -> anyhow::Result<(Vec<f32>, u32, u32)> {
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

        Ok((image_array, original_width, original_height))
    }

    fn generate(&mut self, mut state: GenerationState<'_>) -> anyhow::Result<Vec<i64>> {
        let mut generated = Vec::new();
        let max_steps = generation_step_limit(
            state.input_ids.len(),
            self.max_new_tokens,
            self.input_metadata.fixed_sequence_length,
        )?;

        if max_steps < self.max_new_tokens {
            warn!(
                "ONNX graph fixed sequence length limits generation worker_id={} prompt_tokens={} requested_max_new_tokens={} effective_max_new_tokens={}",
                self.id,
                state.input_ids.len(),
                self.max_new_tokens,
                max_steps
            );
        }

        for _ in 0..max_steps {
            let position = state
                .input_ids
                .len()
                .checked_sub(1)
                .ok_or_else(|| anyhow!("input_ids cannot be empty"))?;
            let logits = self.run_model(&state)?;
            let next_token_id = argmax_token_at_position(&logits, position)?;

            state.input_ids.push(next_token_id);
            state.images_seq_mask.push(false);
            generated.push(next_token_id);

            if next_token_id == EOS_TOKEN_ID {
                break;
            }
        }

        Ok(generated)
    }

    fn run_model(&mut self, state: &GenerationState<'_>) -> anyhow::Result<TensorData> {
        let feeds = make_feeds(
            &self.input_metadata,
            self.image_size,
            FeedInputs {
                input_ids: &state.input_ids,
                images_seq_mask: &state.images_seq_mask,
                image_array: state.image_array,
            },
        )
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

    fn output_metadata(&self, image_array: &[f32], generated_tokens: usize) -> Vec<TensorMetadata> {
        vec![
            tensor_metadata_f32(
                "images_ori",
                &Shape::from([
                    1_i64,
                    3,
                    i64::from(self.image_size),
                    i64::from(self.image_size),
                ]),
                image_array,
            ),
            TensorMetadata {
                name: "generated_token_ids".to_string(),
                shape: vec![1, generated_tokens as i64],
                elements: generated_tokens,
                mean: None,
                min: None,
                max: None,
            },
        ]
    }
}

fn generation_step_limit(
    prompt_tokens: usize,
    requested_max_new_tokens: usize,
    fixed_sequence_length: Option<usize>,
) -> anyhow::Result<usize> {
    let Some(fixed_sequence_length) = fixed_sequence_length else {
        return Ok(requested_max_new_tokens);
    };

    if prompt_tokens > fixed_sequence_length {
        return Err(anyhow!(
            "prompt uses {prompt_tokens} tokens, but the ONNX graph is fixed to {fixed_sequence_length}; shorten text_input or use a model exported with a longer sequence length"
        ));
    }

    Ok(requested_max_new_tokens.min(fixed_sequence_length - prompt_tokens + 1))
}

impl<'a> GenerationState<'a> {
    fn new(prompt_inputs: PromptInputs, image_array: &'a [f32]) -> Self {
        Self {
            input_ids: prompt_inputs.input_ids,
            images_seq_mask: prompt_inputs.images_seq_mask,
            image_array,
        }
    }
}

fn detected_devices(worker_id: usize) -> anyhow::Result<Vec<String>> {
    let env = Environment::current().context("failed to initialize ONNX Runtime environment")?;
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
            worker_id, devices
        );
    }

    Ok(devices)
}

fn generation_metadata(
    prompt_text: String,
    generated_text: String,
    generated_tokens: usize,
) -> GenerationMetadata {
    GenerationMetadata {
        task_token: TASK_TOKEN.to_string(),
        prompt_text,
        result: json!({ "text": generated_text }),
        generated_text,
        generated_tokens,
    }
}

#[cfg(test)]
mod tests;
