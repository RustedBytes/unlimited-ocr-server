use std::{env, path::PathBuf};

use anyhow::anyhow;

use super::defaults::DEFAULT_EXECUTION_PROVIDER;

pub(super) fn env_path(key: &str) -> Option<PathBuf> {
    env::var_os(key).map(PathBuf::from)
}

pub(super) struct SettingSource<T> {
    key: &'static str,
    file_value: Option<T>,
}

impl<T> SettingSource<T> {
    pub(super) fn new(key: &'static str, file_value: Option<T>) -> Self {
        Self { key, file_value }
    }
}

pub(super) fn path_setting(source: SettingSource<PathBuf>, default: &str) -> PathBuf {
    env_path(source.key)
        .or(source.file_value)
        .unwrap_or_else(|| PathBuf::from(default))
}

pub(super) fn string_setting(source: SettingSource<String>, default: &str) -> String {
    env::var(source.key)
        .ok()
        .or(source.file_value)
        .unwrap_or_else(|| default.into())
}

pub(super) fn usize_setting(source: SettingSource<usize>, default: usize) -> anyhow::Result<usize> {
    match env::var(source.key) {
        Ok(value) => value
            .parse()
            .map_err(|err| anyhow!("{} has invalid value `{value}`: {err}", source.key)),
        Err(_) => Ok(source.file_value.unwrap_or(default)),
    }
}

pub(super) fn u64_setting(source: SettingSource<u64>, default: u64) -> anyhow::Result<u64> {
    match env::var(source.key) {
        Ok(value) => value
            .parse()
            .map_err(|err| anyhow!("{} has invalid value `{value}`: {err}", source.key)),
        Err(_) => Ok(source.file_value.unwrap_or(default)),
    }
}

pub(super) fn u32_setting(source: SettingSource<u32>, default: u32) -> anyhow::Result<u32> {
    match env::var(source.key) {
        Ok(value) => value
            .parse()
            .map_err(|err| anyhow!("{} has invalid value `{value}`: {err}", source.key)),
        Err(_) => Ok(source.file_value.unwrap_or(default)),
    }
}

pub(super) fn non_negative_f32_setting(
    source: SettingSource<f32>,
    default: f32,
) -> anyhow::Result<f32> {
    match env::var(source.key) {
        Ok(value) => parse_non_negative_f32(source.key, &value),
        Err(_) => {
            let value = source.file_value.unwrap_or(default);
            validate_non_negative_f32(source.key, value)
        }
    }
}

fn parse_non_negative_f32(key: &str, value: &str) -> anyhow::Result<f32> {
    let parsed = value
        .parse()
        .map_err(|err| anyhow!("{key} has invalid value `{value}`: {err}"))?;

    validate_non_negative_f32(key, parsed)
}

fn validate_non_negative_f32(key: &str, value: f32) -> anyhow::Result<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(anyhow!(
            "{key} has invalid value `{value}`; expected a finite number 0 or greater"
        ));
    }

    Ok(value)
}

pub(super) fn optional_non_negative_i32_setting(
    source: SettingSource<i32>,
) -> anyhow::Result<Option<i32>> {
    match env::var(source.key) {
        Ok(value) => parse_non_negative_i32(source.key, &value).map(Some),
        Err(_) => source
            .file_value
            .map(|value| validate_non_negative_i32(source.key, value))
            .transpose(),
    }
}

fn parse_non_negative_i32(key: &str, value: &str) -> anyhow::Result<i32> {
    let parsed = value
        .parse()
        .map_err(|err| anyhow!("{key} has invalid value `{value}`: {err}"))?;

    validate_non_negative_i32(key, parsed)
}

fn validate_non_negative_i32(key: &str, value: i32) -> anyhow::Result<i32> {
    if value < 0 {
        return Err(anyhow!(
            "{key} has invalid value `{value}`; expected 0 or greater"
        ));
    }

    Ok(value)
}

pub(super) fn bool_setting(source: SettingSource<bool>, default: bool) -> anyhow::Result<bool> {
    match env::var(source.key) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(anyhow!(
                "{} has invalid value `{value}`; expected true or false",
                source.key
            )),
        },
        Err(_) => Ok(source.file_value.unwrap_or(default)),
    }
}

pub(super) fn path_list_setting(key: &str, file_value: Option<Vec<PathBuf>>) -> Vec<PathBuf> {
    env::var_os(key)
        .map(|value| {
            env::split_paths(&value)
                .filter(|path| !path.as_os_str().is_empty())
                .collect::<Vec<_>>()
        })
        .or(file_value)
        .unwrap_or_default()
}

pub(super) fn string_list_setting(key: &str, file_value: Option<Vec<String>>) -> Vec<String> {
    env::var(key)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
        })
        .or(file_value)
        .unwrap_or_default()
}

pub(super) fn secret_setting(key: &str, file_value: Option<String>) -> Option<String> {
    env::var(key)
        .ok()
        .or(file_value)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(super) fn execution_providers_setting(file_value: Option<Vec<String>>) -> Vec<String> {
    // Keep provider names normalized once here; inference can then match simple
    // strings without accepting every spelling variant again.
    let values = env::var("EXECUTION_PROVIDERS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::to_string)
                .collect::<Vec<String>>()
        })
        .or(file_value)
        .unwrap_or_else(|| vec![DEFAULT_EXECUTION_PROVIDER.to_string()]);

    let normalized = values
        .into_iter()
        .map(|value| value.trim().to_ascii_lowercase().replace(['-', '_'], ""))
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();

    if normalized.is_empty() {
        vec![DEFAULT_EXECUTION_PROVIDER.to_string()]
    } else {
        normalized
    }
}
