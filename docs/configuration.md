# Configuration

Copy the sample TOML config:

```bash
cp config.example.toml config.toml
```

The server reads `config.toml` by default. Use another file with:

```bash
cargo run -- --config /path/to/config.toml
```

`CONFIG_PATH=/path/to/config.toml cargo run` is also supported when no `--config` argument is provided.

Sample config:

```toml
[server]
bind_addr = "127.0.0.1:3000"
data_dir = "data"
allow_local_paths = false
local_path_roots = []
cors_allowed_origins = []
api_keys = []
rate_limit_requests_per_minute = 0

[model]
variant = "unlimited_ocr"
image_size = 1024
# path = "Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx"

[queue]
model_pool_size = 1
queue_size = 128
body_limit_bytes = 104857600
request_timeout_seconds = 60

[retention]
job_retention_limit = 1000
metadata_retention_limit = 10000

[validation]
max_image_width = 8192
max_image_height = 8192
max_pdf_pages = 32
pdf_render_dpi = 200

[generation]
max_new_tokens = 256
job_timeout_seconds = 300
webhook_timeout_seconds = 10
webhook_connect_timeout_seconds = 5
webhook_max_attempts = 1
webhook_initial_backoff_ms = 500
webhook_signing_secret = ""
allow_private_webhook_urls = false

[runtime]
execution_providers = ["auto"]
# device_id = 0

[logging]
rust_log = "info,ort=warn"
```

Supported `model.variant` values:

- `unlimited_ocr`
- `custom`

`model.image_size` controls the square image tensor size used for preprocessing. It defaults to `1024` and must match a fixed `images_ori` graph dimension when the ONNX graph encodes one.

Supported `runtime.execution_providers` values:

- `auto`: ORT auto-device policy, with CPU fallback
- `cpu`: CPU only
- `coreml`: CoreML with all Apple compute units
- `coreml_gpu`: CoreML CPU+GPU
- `coreml_npu`: CoreML CPU+Neural Engine
- `xnnpack`: XNNPACK when available in the ORT build
- `cuda`: NVIDIA CUDA, only when built with the Cargo `cuda` feature

`auto` is the default. Explicit providers remain opt-in because provider support depends on the exported ONNX graph and local runtime.

`runtime.device_id` selects the ORT device id for execution providers that support explicit device selection. It is currently honored by CUDA. Omit it to use the provider default device.

Build with CUDA support:

```bash
cargo run --features cuda
```

Then request CUDA explicitly:

```toml
[runtime]
execution_providers = ["cuda", "cpu"]
device_id = 1
```

or with the environment:

```bash
EXECUTION_PROVIDERS=cuda,cpu INFERENCE_DEVICE_ID=1 cargo run --features cuda
```

Environment variables override TOML values when set:

- `BIND_ADDR`: bind address, default `127.0.0.1:3000`
- `DATA_DIR`: image and JSONL metadata directory, default `data`
- `ALLOW_LOCAL_PATHS`: set to `true` to enable `/v1/infer/path`
- `LOCAL_PATH_ROOTS`: platform-separated allowed roots for `/v1/infer/path`
- `CORS_ALLOWED_ORIGINS`: comma-separated origins allowed by browser CORS checks
- `API_KEYS`: comma-separated accepted API keys; empty disables API key authentication
- `RATE_LIMIT_REQUESTS_PER_MINUTE`: process-wide request limit; set to `0` to disable rate limiting
- `MODEL_POOL_SIZE`: number of model workers
- `QUEUE_SIZE`: queued job capacity
- `BODY_LIMIT_BYTES`: whole HTTP request body limit for multipart uploads, default `104857600`
- `REQUEST_TIMEOUT_SECONDS`: whole HTTP request timeout; set to `0` to disable timeout enforcement
- `JOB_RETENTION_LIMIT`: maximum in-memory job records kept queryable through `/v1/jobs/{id}`
- `METADATA_RETENTION_LIMIT`: maximum latest metadata records kept when JSONL files are compacted at startup; set to `0` to disable compaction
- `MAX_IMAGE_WIDTH`: maximum accepted image width before decode
- `MAX_IMAGE_HEIGHT`: maximum accepted image height before decode
- `MAX_PDF_PAGES`: maximum accepted PDF page count
- `PDF_RENDER_DPI`: DPI used when rendering PDF pages to PNG before OCR
- `MODEL_VARIANT`: model variant
- `MODEL_PATH`: explicit ONNX model path, overrides `MODEL_VARIANT` path selection
- `DECODE_MODEL_PATH`: optional text decode ONNX path for KV-cache generation
- `MODEL_IMAGE_SIZE`: square image tensor size, default `1024`
- `MAX_NEW_TOKENS`: maximum decoder tokens per generation
- `JOB_TIMEOUT_SECONDS`: per-job inference timeout; set to `0` to disable timeout enforcement
- `WEBHOOK_TIMEOUT_SECONDS`: total outbound webhook request timeout; set to `0` to disable
- `WEBHOOK_CONNECT_TIMEOUT_SECONDS`: outbound webhook connection timeout; set to `0` to disable
- `WEBHOOK_MAX_ATTEMPTS`: maximum webhook delivery attempts, including the first attempt
- `WEBHOOK_INITIAL_BACKOFF_MS`: initial webhook retry backoff; later retries double this delay
- `WEBHOOK_SIGNING_SECRET`: HMAC-SHA256 signing secret for webhook bodies; empty disables signatures
- `ALLOW_PRIVATE_WEBHOOK_URLS`: set to `true` only in trusted deployments that must call local or private webhook targets
- `EXECUTION_PROVIDERS`: comma-separated provider list, for example `auto` or `coreml,auto`
- `INFERENCE_DEVICE_ID`: non-negative ORT device id for providers that support explicit device selection, currently CUDA
- `RUST_LOG`: logging level, for example `debug`
- `CONFIG_PATH`: explicit TOML config path when `--config` is not set

`/v1/infer/path` is disabled by default. To enable it safely:

```toml
[server]
allow_local_paths = true
local_path_roots = ["/srv/unlimited-ocr-inputs"]
```

Only files under the configured roots are accepted. The server rejects startup configuration where local paths are enabled without at least one root.

CORS response headers are disabled by default. Configure explicit allowed origins for browser clients:

```toml
[server]
cors_allowed_origins = ["http://localhost:5173"]
```

Only `GET`, `POST`, and `OPTIONS` methods are allowed by the CORS layer.

API key authentication is disabled by default. Configure one or more keys to protect all endpoints except `/health` and `/ready`:

```toml
[server]
api_keys = ["replace-with-a-long-random-secret"]
```

Clients can send the key with either `x-api-key` or a bearer token:

```bash
curl -H 'x-api-key: replace-with-a-long-random-secret' http://127.0.0.1:3000/metrics
```

Rate limiting is also disabled by default. When `server.rate_limit_requests_per_minute` is greater than `0`, the server applies a process-wide fixed-window limit to all endpoints except `/health` and `/ready`. Exceeded requests return `429 Too Many Requests` with `Retry-After`.

Request validation happens before the image is decoded for inference:

- `queue.body_limit_bytes` limits the whole multipart request body. Keep it slightly above the largest file size you want to accept so multipart headers and fields fit too.
- `queue.request_timeout_seconds` limits total HTTP request handling time and returns `408 Request Timeout`.
- `validation.max_image_width` and `validation.max_image_height` reject oversized image dimensions.
- `validation.max_pdf_pages` rejects PDFs with too many pages before rendering.
- `validation.pdf_render_dpi` controls the PNG render resolution for PDF pages.
- Only supported image formats with `image/*` content types and PDFs are accepted.

Jobs are marked failed if inference exceeds `generation.job_timeout_seconds`. A timed-out blocking inference task may finish in the background, so the worker slot is restarted before it accepts more work.

Webhook delivery uses `generation.webhook_timeout_seconds` for the full request and `generation.webhook_connect_timeout_seconds` for establishing the connection. Failed webhook deliveries retry up to `generation.webhook_max_attempts` times with exponential backoff starting at `generation.webhook_initial_backoff_ms`. Final failures are appended to `data/metadata/webhooks_dead_letter.jsonl`. Webhook failures are logged and do not change the completed job result.

When `generation.webhook_signing_secret` is set, webhook requests include the legacy compatibility header `x-server-signature: sha256=<hex-hmac>`, computed over the exact JSON request body. Webhook requests also include `x-server-event-id`, `x-server-event-type`, and `x-server-delivery-attempt`.

For SSRF protection, webhook URLs reject credentials, fragments, localhost, and literal private/local IP addresses by default. Redirects are not followed. If private webhook targets are required in a trusted network, set `generation.allow_private_webhook_urls = true`.
