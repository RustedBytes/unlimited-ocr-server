use std::path::{Path, PathBuf};

use anyhow::anyhow;
use log::warn;
use ort::{
    ep::{self, ExecutionProviderDispatch},
    session::{
        Session,
        builder::{AutoDevicePolicy, GraphOptimizationLevel},
    },
};

pub(super) fn tokenizer_path_for_model(model_path: &Path) -> anyhow::Result<PathBuf> {
    let onnx_dir = model_path.parent().ok_or_else(|| {
        anyhow!(
            "model path has no parent directory: {}",
            model_path.display()
        )
    })?;
    let model_dir = onnx_dir.parent().unwrap_or(onnx_dir);

    for path in [
        model_dir.join("tokenizer.json"),
        onnx_dir.join("tokenizer.json"),
    ] {
        if path.exists() {
            return Ok(path);
        }
    }

    Ok(model_dir.join("tokenizer.json"))
}

pub(super) fn load_session(path: &Path, execution_providers: &[String]) -> anyhow::Result<Session> {
    let requested_eps = execution_provider_dispatches(execution_providers);
    let use_auto_device = execution_providers
        .iter()
        .any(|provider| matches!(provider.as_str(), "auto" | "autodevice"));
    let cpu_only = execution_providers
        .iter()
        .all(|provider| provider.as_str() == "cpu");
    let has_requested_eps = !requested_eps.is_empty();

    // Keep explicit EP registration opt-in. Some providers can register but
    // still spend a long time compiling unsupported dynamic graph segments.
    let mut builder = Session::builder()
        .map_err(|err| anyhow!("failed to create ONNX session builder: {err}"))?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|err| anyhow!("failed to set ONNX graph optimization level: {err}"))?
        .with_prepacking(true)
        .map_err(|err| anyhow!("failed to enable ONNX prepacking: {err}"))?
        .with_memory_pattern(false)
        .map_err(|err| anyhow!("failed to configure ONNX memory pattern optimization: {err}"))?;

    if execution_providers
        .iter()
        .any(|provider| provider.as_str() == "cuda")
    {
        builder = builder
            .with_device_allocated_initializers()
            .map_err(|err| anyhow!("failed to enable ONNX device-allocated initializers: {err}"))?;
    }

    if has_requested_eps {
        builder = builder
            .with_execution_providers(&requested_eps)
            .map_err(|err| {
                anyhow!(
                    "failed to register ONNX execution providers for {}: {err}",
                    path.display()
                )
            })?;
    }

    if use_auto_device || (!has_requested_eps && !cpu_only) {
        builder = builder
            .with_auto_device(AutoDevicePolicy::MaxPerformance)
            .map_err(|err| anyhow!("failed to enable ONNX Runtime auto device selection: {err}"))?;
    }

    builder
        .commit_from_file(path)
        .map_err(|err| anyhow!("failed to load ONNX model {}: {err}", path.display()))
}

pub(super) fn execution_provider_dispatches(
    execution_providers: &[String],
) -> Vec<ExecutionProviderDispatch> {
    execution_providers
        .iter()
        .filter_map(|provider| execution_provider_dispatch(provider))
        .collect()
}

fn execution_provider_dispatch(provider: &str) -> Option<ExecutionProviderDispatch> {
    match provider {
        "coreml" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::All,
            true,
        )),
        "coremlgpu" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::CPUAndGPU,
            true,
        )),
        "coremlnpu" | "coremlane" | "ane" | "npu" => Some(coreml_execution_provider(
            ep::coreml::ComputeUnits::CPUAndNeuralEngine,
            false,
        )),
        "cuda" => cuda_execution_provider(),
        "xnnpack" => Some(ep::XNNPACK::default().build()),
        "auto" | "autodevice" | "cpu" => None,
        other => {
            warn!("unknown execution provider `{other}` ignored");
            None
        }
    }
}

fn coreml_execution_provider(
    compute_units: ep::coreml::ComputeUnits,
    low_precision_accumulation_on_gpu: bool,
) -> ExecutionProviderDispatch {
    ep::CoreML::default()
        .with_compute_units(compute_units)
        .with_model_format(ep::coreml::ModelFormat::MLProgram)
        .with_low_precision_accumulation_on_gpu(low_precision_accumulation_on_gpu)
        .build()
}

#[cfg(feature = "cuda")]
fn cuda_execution_provider() -> Option<ExecutionProviderDispatch> {
    Some(
        ep::CUDA::default()
            .with_tf32(true)
            .with_prefer_nhwc(true)
            .with_conv_max_workspace(true)
            .build(),
    )
}

#[cfg(not(feature = "cuda"))]
fn cuda_execution_provider() -> Option<ExecutionProviderDispatch> {
    warn!("CUDA execution provider requested but this binary was built without `--features cuda`");
    None
}
