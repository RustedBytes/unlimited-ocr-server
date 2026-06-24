use std::{collections::HashMap, io::ErrorKind, path::Path};

use anyhow::Context;
use uuid::Uuid;

use crate::{
    config::Config,
    types::{JobRecord, JobStatus},
    util::append_jsonl,
};

pub async fn load_jobs(config: &Config) -> anyhow::Result<HashMap<Uuid, JobRecord>> {
    let mut jobs = HashMap::new();
    load_job_records(&config.submissions_jsonl, &mut jobs).await?;
    load_job_records(&config.results_jsonl, &mut jobs).await?;

    let mut recovered = Vec::new();
    for record in jobs.values_mut() {
        if matches!(record.status, JobStatus::Queued | JobStatus::Running) {
            record.status = JobStatus::Failed;
            record.updated_at = time::OffsetDateTime::now_utc();
            record.error = Some("server restarted before job reached a terminal state".to_string());
            recovered.push(record.clone());
        }
    }

    for record in recovered {
        append_jsonl(&config.results_jsonl, &record)
            .await
            .with_context(|| {
                format!(
                    "failed to append recovered job state to {}",
                    config.results_jsonl.display()
                )
            })?;
    }

    compact_metadata(config, &jobs).await?;
    retain_recent_jobs(&mut jobs, config.job_retention_limit);

    Ok(jobs)
}

async fn load_job_records(path: &Path, jobs: &mut HashMap<Uuid, JobRecord>) -> anyhow::Result<()> {
    let contents = match tokio::fs::read_to_string(path).await {
        Ok(contents) => contents,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(err).with_context(|| format!("failed to read {}", path.display()));
        }
    };

    for (line_number, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<JobRecord>(line).with_context(|| {
            format!(
                "failed to parse job record at {}:{}",
                path.display(),
                line_number + 1
            )
        })?;
        jobs.insert(record.id, record);
    }

    Ok(())
}

async fn compact_metadata(config: &Config, jobs: &HashMap<Uuid, JobRecord>) -> anyhow::Result<()> {
    if config.metadata_retention_limit == 0 {
        return Ok(());
    }

    let records = recent_records(jobs, config.metadata_retention_limit);
    let (submissions, results): (Vec<_>, Vec<_>) = records
        .into_iter()
        .partition(|record| matches!(record.status, JobStatus::Queued | JobStatus::Running));

    write_jsonl_records(&config.submissions_jsonl, &submissions).await?;
    write_jsonl_records(&config.results_jsonl, &results).await?;
    Ok(())
}

async fn write_jsonl_records(path: &Path, records: &[JobRecord]) -> anyhow::Result<()> {
    let temp_path = path.with_extension("jsonl.tmp");
    let mut bytes = Vec::with_capacity(records.len().saturating_mul(256));
    for record in records {
        serde_json::to_writer(&mut bytes, record)?;
        bytes.push(b'\n');
    }
    tokio::fs::write(&temp_path, bytes)
        .await
        .with_context(|| format!("failed to write {}", temp_path.display()))?;
    tokio::fs::rename(&temp_path, path)
        .await
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

pub(super) fn retain_recent_jobs(jobs: &mut HashMap<Uuid, JobRecord>, limit: usize) {
    if jobs.len() <= limit {
        return;
    }
    if limit == 0 {
        jobs.clear();
        return;
    }

    let keep = recent_records(jobs, limit)
        .into_iter()
        .map(|record| record.id)
        .collect::<std::collections::HashSet<_>>();
    jobs.retain(|id, _| keep.contains(id));
}

fn recent_records(jobs: &HashMap<Uuid, JobRecord>, limit: usize) -> Vec<JobRecord> {
    let mut records = jobs.values().cloned().collect::<Vec<_>>();
    records.sort_by_key(|record| record.updated_at);
    records.reverse();
    records.truncate(limit);
    records.reverse();
    records
}
