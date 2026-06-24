use image::{DynamicImage, Rgb, RgbImage};

use super::model::execution_provider_dispatches;
use super::*;

#[test]
fn computes_unlimited_ocr_image_token_count() {
    assert_eq!(image_token_count(1024), 273);
    assert_eq!(image_token_count(640), 111);
}

#[test]
fn builds_prompt_with_implicit_image_token() {
    let tokenizer = test_tokenizer();
    let prompt = build_image_prompt(&tokenizer, "document parsing.", 1024).unwrap();

    assert_eq!(prompt.input_ids[0], BOS_TOKEN_ID);
    assert_eq!(prompt.images_seq_mask.len(), prompt.input_ids.len());
    assert_eq!(
        prompt
            .images_seq_mask
            .iter()
            .filter(|value| **value)
            .count(),
        image_token_count(1024)
    );
    assert!(
        prompt.input_ids[1..=image_token_count(1024)]
            .iter()
            .all(|id| *id == IMAGE_TOKEN_ID)
    );
}

#[test]
fn builds_prompt_with_explicit_image_token() {
    let tokenizer = test_tokenizer();
    let prompt = build_image_prompt(&tokenizer, "read <image> now", 640).unwrap();

    assert_eq!(prompt.input_ids[0], BOS_TOKEN_ID);
    assert!(!prompt.images_seq_mask[1]);
    assert_eq!(
        prompt
            .images_seq_mask
            .iter()
            .filter(|value| **value)
            .count(),
        image_token_count(640)
    );
}

#[test]
fn pads_fixed_length_inputs() {
    let values = prepare_i64_1d(&[1, 2, 3], Some(5), 0, "input_ids").unwrap();

    assert_eq!(values, vec![1, 2, 3, 0, 0]);
}

#[test]
fn rejects_inputs_longer_than_fixed_length() {
    let err = prepare_bool_1d(&[true, false, true], Some(2), false, "images_seq_mask").unwrap_err();

    assert!(
        err.to_string()
            .contains("input `images_seq_mask` has 3 tokens")
    );
}

#[test]
fn preprocesses_image_with_gray_padding_and_chw_normalization() {
    let mut image = RgbImage::from_pixel(2, 1, Rgb([255, 0, 0]));
    image.put_pixel(1, 0, Rgb([0, 255, 0]));

    let got = preprocess_image(DynamicImage::ImageRgb8(image), 1024).unwrap();
    let plane = 1024 * 1024;

    assert_eq!(got.len(), 3 * plane);
    assert!((got[0] - -0.0039215684).abs() < 0.000001);
    let first_image_pixel = 256 * 1024;
    assert_eq!(got[first_image_pixel], 1.0);
    assert_eq!(got[plane + first_image_pixel], -1.0);
    assert_eq!(got[2 * plane + first_image_pixel], -1.0);
}

#[test]
fn cleans_trailing_unlimited_ocr_eos_text() {
    let got = clean_generated_text("invoice total<｜end▁of▁sentence｜>");

    assert_eq!(got, "invoice total");
}

#[test]
fn cuda_provider_is_feature_gated() {
    let providers = vec!["cuda".to_string()];
    let dispatches = execution_provider_dispatches(&providers);

    assert_eq!(dispatches.len(), usize::from(cfg!(feature = "cuda")));
}

fn test_tokenizer() -> Tokenizer {
    Tokenizer::from_file("Unlimited-OCR/tokenizer.json").unwrap()
}
