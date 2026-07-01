use anyhow::anyhow;
use tokenizers::Tokenizer;

use crate::types::TaskSpec;

pub(super) const DEFAULT_PROMPT: &str = "<|grounding|><image>Convert the document to markdown.";
pub(super) const IMAGE_TOKEN: &str = "<image>";
pub(super) const BOS_TOKEN_ID: i64 = 0;
pub(super) const EOS_TOKEN_ID: i64 = 1;
pub(super) const IMAGE_TOKEN_ID: i64 = 128_815;

const PATCH_SIZE: u32 = 16;
const DOWNSAMPLE_RATIO: u32 = 4;

#[derive(Debug)]
pub(super) struct PromptInputs {
    pub(super) input_ids: Vec<i64>,
    pub(super) images_seq_mask: Vec<bool>,
}

pub(super) fn prompt_from_task(task: &TaskSpec) -> String {
    task.text_input
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_PROMPT)
        .to_string()
}

pub(super) fn build_image_prompt(
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

pub(super) fn image_token_count(image_size: u32) -> usize {
    let image_patches = image_size / PATCH_SIZE;
    let num_queries = image_patches.div_ceil(DOWNSAMPLE_RATIO);
    ((num_queries + 1) * num_queries + 1) as usize
}

pub(super) fn decode_generated_text(
    tokenizer: &Tokenizer,
    generated_ids: &[i64],
) -> anyhow::Result<String> {
    let generated_u32 = generated_ids
        .iter()
        .filter_map(|id| u32::try_from(*id).ok())
        .collect::<Vec<_>>();
    let raw_text = tokenizer
        .decode(&generated_u32, false)
        .map_err(|err| anyhow!("failed to decode generated tokens: {err}"))?;
    Ok(clean_generated_text(&raw_text))
}

pub(super) fn clean_generated_text(raw: &str) -> String {
    raw.strip_suffix("<｜end▁of▁sentence｜>")
        .unwrap_or(raw)
        .trim()
        .to_string()
}
