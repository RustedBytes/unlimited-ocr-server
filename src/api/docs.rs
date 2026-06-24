use axum::http::{HeaderName, HeaderValue, Method, header};
use log::warn;
use serde_json::Value;
use tower_http::cors::{AllowOrigin, CorsLayer};

pub(super) fn openapi_document() -> Value {
    let mut document: Value = serde_json::from_str(include_str!("../../docs/openapi.json"))
        .expect("embedded OpenAPI document must be valid JSON");
    document["info"]["version"] = Value::String(env!("CARGO_PKG_VERSION").to_string());
    document
}

pub(super) fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let origins = allowed_origins
        .iter()
        .filter_map(|origin| match HeaderValue::from_str(origin) {
            Ok(origin) => Some(origin),
            Err(err) => {
                warn!("invalid CORS origin ignored origin={origin:?} error={err}");
                None
            }
        })
        .collect::<Vec<_>>();

    CorsLayer::new()
        .allow_origin(AllowOrigin::list(origins))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            header::CONTENT_TYPE,
            header::AUTHORIZATION,
            HeaderName::from_static("x-api-key"),
        ])
}
