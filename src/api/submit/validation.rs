use std::{
    io::Cursor,
    net::{IpAddr, Ipv4Addr, Ipv6Addr},
};

use image::ImageReader;
use log::warn;

use crate::{config::Config, state::AppState, util::image_format_content_type};

use super::super::ApiError;

pub(in crate::api) fn validate_webhook_url(
    config: &Config,
    webhook_url: Option<String>,
) -> Result<Option<String>, ApiError> {
    let Some(webhook_url) = webhook_url.map(|value| value.trim().to_string()) else {
        return Ok(None);
    };
    if webhook_url.is_empty() {
        return Ok(None);
    }

    let parsed = reqwest::Url::parse(&webhook_url)
        .map_err(|err| ApiError::BadRequest(format!("invalid webhook_url: {err}")))?;
    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(ApiError::BadRequest(format!(
                "unsupported webhook_url scheme `{scheme}`; expected http or https"
            )));
        }
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(ApiError::BadRequest(
            "webhook_url must not include credentials".to_string(),
        ));
    }
    if parsed.fragment().is_some() {
        return Err(ApiError::BadRequest(
            "webhook_url must not include a fragment".to_string(),
        ));
    }
    if !config.allow_private_webhook_urls {
        validate_public_webhook_host(&parsed)?;
    }
    Ok(Some(parsed.to_string()))
}

fn validate_public_webhook_host(url: &reqwest::Url) -> Result<(), ApiError> {
    let Some(host) = url.host_str() else {
        return Err(ApiError::BadRequest(
            "webhook_url must include a host".to_string(),
        ));
    };

    if let Ok(ip) = host.parse::<IpAddr>() {
        if ip_is_private_or_local(ip) {
            return Err(private_webhook_url_error());
        }
    } else {
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        if host == "localhost" || host.ends_with(".localhost") {
            return Err(private_webhook_url_error());
        }
    }

    Ok(())
}

fn ip_is_private_or_local(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => ipv4_is_private_or_local(ip),
        IpAddr::V6(ip) => ipv6_is_private_or_local(ip),
    }
}

fn private_webhook_url_error() -> ApiError {
    ApiError::BadRequest(
        "webhook_url targets a private or local address; set allow_private_webhook_urls for trusted deployments".to_string(),
    )
}

fn ipv4_is_private_or_local(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.is_unspecified()
        || ip.octets()[0] == 0
}

fn ipv6_is_private_or_local(ip: Ipv6Addr) -> bool {
    ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || matches!(ip.to_ipv4_mapped().map(IpAddr::V4), Some(IpAddr::V4(ip)) if ipv4_is_private_or_local(ip))
}

pub(in crate::api) fn ensure_workers_ready(state: &AppState) -> Result<(), ApiError> {
    if state.workers.is_ready() {
        Ok(())
    } else {
        Err(ApiError::ServiceUnavailable(
            "model workers are not ready".to_string(),
        ))
    }
}

pub(in crate::api) async fn ensure_local_path_allowed(
    state: &AppState,
    image_path: &std::path::Path,
) -> Result<(), ApiError> {
    if !state.config.allow_local_paths {
        return Err(ApiError::Forbidden(
            "local path inference is disabled".to_string(),
        ));
    }

    for root in &state.config.local_path_roots {
        match tokio::fs::canonicalize(root).await {
            Ok(root) if image_path.starts_with(&root) && image_path != root => return Ok(()),
            Ok(_) => {}
            Err(err) => {
                warn!(
                    "configured local path root could not be resolved root={} error={}",
                    root.display(),
                    err
                );
            }
        }
    }

    Err(ApiError::Forbidden(format!(
        "image path is outside configured local path roots: {}",
        image_path.display()
    )))
}

pub(in crate::api) fn validate_image_bytes(
    state: &AppState,
    content_type: Option<&str>,
    bytes: &[u8],
) -> Result<(), ApiError> {
    if let Some(content_type) = content_type
        && !content_type.starts_with("image/")
    {
        return Err(ApiError::BadRequest(format!(
            "unsupported content type `{content_type}`; expected an image"
        )));
    }

    let format = image::guess_format(bytes)
        .map_err(|_| ApiError::BadRequest("unsupported or invalid image format".to_string()))?;
    if image_format_content_type(format).is_none() {
        return Err(ApiError::BadRequest(format!(
            "unsupported image format: {format:?}"
        )));
    }

    let reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|err| ApiError::BadRequest(format!("failed to identify image format: {err}")))?;
    let (width, height) = reader
        .into_dimensions()
        .map_err(|err| ApiError::BadRequest(format!("failed to read image dimensions: {err}")))?;

    if width > state.config.max_image_width || height > state.config.max_image_height {
        return Err(ApiError::BadRequest(format!(
            "image dimensions {}x{} exceed configured limit {}x{}",
            width, height, state.config.max_image_width, state.config.max_image_height
        )));
    }

    Ok(())
}
