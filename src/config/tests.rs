use std::{
    env, fs,
    time::{SystemTime, UNIX_EPOCH},
};

use super::*;
use super::{
    file::FileConfig,
    model_variant::parse_model_variant_value,
    settings::{
        SettingSource, execution_providers_setting, optional_non_negative_i32_setting,
        string_list_setting,
    },
};

#[test]
fn parses_supported_model_variant_aliases() {
    let cases = [
        ("unlimited_ocr", "unlimited_ocr"),
        ("Unlimited-OCR", "unlimited_ocr"),
        ("default", "unlimited_ocr"),
        ("custom", "custom"),
    ];

    for (raw, expected) in cases {
        let variant = parse_model_variant_value(raw).unwrap();
        assert_eq!(variant.as_str(), expected);
    }
}

#[test]
fn rejects_unknown_model_variant() {
    let err = parse_model_variant_value("fp16").unwrap_err();

    assert!(err.to_string().contains("unsupported model variant `fp16`"));
}

#[test]
fn normalizes_execution_provider_names() {
    let providers = execution_providers_setting(Some(vec![
        " CoreML-GPU ".to_string(),
        "CUDA".to_string(),
        "xnn_pack".to_string(),
    ]));

    assert_eq!(providers, vec!["coremlgpu", "cuda", "xnnpack"]);
}

#[test]
fn cuda_execution_provider_defaults_to_one_worker() {
    let workers = default_worker_count_for_execution_providers(&["cuda".to_string()]);

    assert_eq!(workers, 1);
}

#[test]
fn cpu_execution_provider_uses_parallel_worker_default() {
    let workers = default_worker_count_for_execution_providers(&["cpu".to_string()]);

    assert!((1..=4).contains(&workers));
}

#[test]
fn rejects_negative_inference_device_id() {
    let err = optional_non_negative_i32_setting(SettingSource::new(
        "INFERENCE_DEVICE_ID_UNUSED_IN_TEST",
        Some(-1),
    ))
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("INFERENCE_DEVICE_ID_UNUSED_IN_TEST has invalid value `-1`")
    );
}

#[test]
fn keeps_configured_cors_origins() {
    let origins = string_list_setting(
        "CORS_ALLOWED_ORIGINS_UNUSED_IN_TEST",
        Some(vec![
            "http://localhost:5173".to_string(),
            "https://example.com".to_string(),
        ]),
    );

    assert_eq!(
        origins,
        vec![
            "http://localhost:5173".to_string(),
            "https://example.com".to_string()
        ]
    );
}

#[test]
fn model_variant_selects_matching_default_model_path() {
    assert_eq!(
        ModelVariant::UnlimitedOcr.default_model_path(),
        PathBuf::from("Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx")
    );
    assert_eq!(
        ModelVariant::Custom.default_model_path(),
        PathBuf::from("Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx")
    );
}

#[test]
fn explicit_missing_config_path_is_an_error() {
    let path = unique_temp_path("missing-config.toml");
    let err = FileConfig::load(Some(path.clone())).unwrap_err();

    assert!(
        err.to_string()
            .contains(&format!("config file does not exist: {}", path.display()))
    );
}

#[test]
fn loads_file_config_from_toml() {
    let path = unique_temp_path("config.toml");
    fs::write(
        &path,
        r#"
[server]
bind_addr = "127.0.0.1:9999"
data_dir = "tmp-data"
api_keys = ["secret"]
rate_limit_requests_per_minute = 120

[model]
variant = "unlimited_ocr"
image_size = 768
decode_path = "Unlimited-OCR/onnx/unlimited_ocr_decode.onnx"

[queue]
model_pool_size = 2
queue_size = 8
body_limit_bytes = 4096
request_timeout_seconds = 20

[retention]
job_retention_limit = 64
metadata_retention_limit = 128

[validation]
max_image_width = 640
max_image_height = 480
max_pdf_pages = 12
pdf_render_dpi = 180

[generation]
max_new_tokens = 32
job_timeout_seconds = 45
webhook_timeout_seconds = 12
webhook_connect_timeout_seconds = 3
webhook_max_attempts = 4
webhook_initial_backoff_ms = 250
webhook_signing_secret = "webhook-secret"
allow_private_webhook_urls = true

[runtime]
execution_providers = ["coreml-gpu", "xnnpack"]
device_id = 1

[logging]
rust_log = "debug"
"#,
    )
    .unwrap();

    let config = FileConfig::load(Some(path.clone())).unwrap();
    fs::remove_file(path).unwrap();

    let server = config.server.unwrap();
    let model = config.model.unwrap();
    let queue = config.queue.unwrap();
    let retention = config.retention.unwrap();
    let validation = config.validation.unwrap();
    let generation = config.generation.unwrap();
    let runtime = config.runtime.unwrap();
    let logging = config.logging.unwrap();

    assert_eq!(server.bind_addr.as_deref(), Some("127.0.0.1:9999"));
    assert_eq!(server.data_dir, Some(PathBuf::from("tmp-data")));
    assert_eq!(server.api_keys, Some(vec!["secret".to_string()]));
    assert_eq!(server.rate_limit_requests_per_minute, Some(120));
    assert_eq!(model.variant.as_deref(), Some("unlimited_ocr"));
    assert_eq!(model.image_size, Some(768));
    assert_eq!(
        model.decode_path,
        Some(PathBuf::from(
            "Unlimited-OCR/onnx/unlimited_ocr_decode.onnx"
        ))
    );
    assert_eq!(queue.model_pool_size, Some(2));
    assert_eq!(queue.queue_size, Some(8));
    assert_eq!(queue.body_limit_bytes, Some(4096));
    assert_eq!(queue.request_timeout_seconds, Some(20));
    assert_eq!(retention.job_retention_limit, Some(64));
    assert_eq!(retention.metadata_retention_limit, Some(128));
    assert_eq!(validation.max_image_width, Some(640));
    assert_eq!(validation.max_image_height, Some(480));
    assert_eq!(validation.max_pdf_pages, Some(12));
    assert_eq!(validation.pdf_render_dpi, Some(180));
    assert_eq!(generation.max_new_tokens, Some(32));
    assert_eq!(generation.job_timeout_seconds, Some(45));
    assert_eq!(generation.webhook_timeout_seconds, Some(12));
    assert_eq!(generation.webhook_connect_timeout_seconds, Some(3));
    assert_eq!(generation.webhook_max_attempts, Some(4));
    assert_eq!(generation.webhook_initial_backoff_ms, Some(250));
    assert_eq!(
        generation.webhook_signing_secret.as_deref(),
        Some("webhook-secret")
    );
    assert_eq!(generation.allow_private_webhook_urls, Some(true));
    assert_eq!(
        runtime.execution_providers,
        Some(vec!["coreml-gpu".to_string(), "xnnpack".to_string()])
    );
    assert_eq!(runtime.device_id, Some(1));
    assert_eq!(logging.rust_log.as_deref(), Some("debug"));
}

#[test]
fn infers_decode_model_path_from_prefill_sibling() {
    let root = unique_temp_path("onnx");
    fs::create_dir_all(&root).unwrap();
    let prefill_path = root.join("unlimited_ocr_prefill.onnx");
    let decode_path = root.join("unlimited_ocr_decode.onnx");
    fs::write(&prefill_path, b"prefill").unwrap();
    fs::write(&decode_path, b"decode").unwrap();

    let got = inferred_decode_model_path(&prefill_path);

    assert_eq!(got, Some(decode_path));
    fs::remove_dir_all(root).unwrap();
}

fn unique_temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    env::temp_dir().join(format!("unlimited-ocr-server-{nanos}-{name}"))
}
