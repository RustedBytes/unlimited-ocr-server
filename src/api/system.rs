use std::{process::Stdio, time::Duration};

use crate::types::GpuMemoryResponse;

const GPU_MEMORY_QUERY_TIMEOUT: Duration = Duration::from_millis(500);

pub(super) async fn system_usage() -> (Option<u64>, Option<GpuMemoryResponse>) {
    let process_memory_rss_bytes = process_memory_rss_bytes();
    let gpu_memory = gpu_memory_usage().await;
    (process_memory_rss_bytes, gpu_memory)
}

#[cfg(target_os = "linux")]
fn process_memory_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        let value = line.strip_prefix("VmRSS:")?.trim();
        let kb = value.split_whitespace().next()?.parse::<u64>().ok()?;
        Some(kb * 1024)
    })
}

#[cfg(not(target_os = "linux"))]
fn process_memory_rss_bytes() -> Option<u64> {
    None
}

async fn gpu_memory_usage() -> Option<GpuMemoryResponse> {
    let output = tokio::time::timeout(
        GPU_MEMORY_QUERY_TIMEOUT,
        tokio::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output(),
    )
    .await
    .ok()?
    .ok()?;

    if !output.status.success() {
        return None;
    }

    parse_nvidia_smi_memory(&String::from_utf8_lossy(&output.stdout))
}

pub(super) fn parse_nvidia_smi_memory(output: &str) -> Option<GpuMemoryResponse> {
    let mut used_mib = 0_u64;
    let mut total_mib = 0_u64;
    for line in output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let (used, total) = line.split_once(',')?;
        used_mib += used.trim().parse::<u64>().ok()?;
        total_mib += total.trim().parse::<u64>().ok()?;
    }

    (total_mib > 0).then_some(GpuMemoryResponse {
        used_bytes: used_mib * 1024 * 1024,
        total_bytes: total_mib * 1024 * 1024,
    })
}
