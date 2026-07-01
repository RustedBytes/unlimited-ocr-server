# HTTP API

Health:

```bash
curl http://127.0.0.1:3000/health
```

Readiness:

```bash
curl http://127.0.0.1:3000/ready
```

`/health` is a liveness endpoint. `/ready` returns `200` only after at least one model worker has initialized successfully; otherwise it returns `503`.

When `server.api_keys` is configured, all endpoints except `/health` and `/ready` require an API key:

```bash
curl -H 'x-api-key: replace-with-a-long-random-secret' http://127.0.0.1:3000/metrics
```

`Authorization: Bearer <key>` is also accepted.

Metrics snapshot:

```bash
curl http://127.0.0.1:3000/metrics
```

The metrics response includes HTTP request counts and latency, queue depth, job/inference counts and latency, worker restarts, model load time, webhook failures, cleanup failures, retained jobs, Linux process RSS memory when available, and aggregate NVIDIA GPU memory when `nvidia-smi` is available.

Human-readable retained jobs table:

```bash
curl 'http://127.0.0.1:3000/jobs?page=1&per_page=25'
```

OpenAPI document:

```bash
curl http://127.0.0.1:3000/openapi.json
```

Upload an image or PDF:

```bash
curl -s \
  -F image=@/path/to/input.pdf \
  -F text_input='<|grounding|><image>Convert the document to markdown.' \
  -F webhook_url='https://example.com/unlimited-ocr-webhook' \
  http://127.0.0.1:3000/v1/infer
```

Image uploads return one queued job. PDF uploads are rendered to PNG pages and return one queued job per page:

```json
{
  "kind": "pdf",
  "page_count": 2,
  "jobs": [
    {
      "id": "9b55eaa1-7f6f-4ac6-89ef-16e6b2f9b927",
      "status": "queued",
      "status_url": "/v1/jobs/9b55eaa1-7f6f-4ac6-89ef-16e6b2f9b927"
    }
  ]
}
```

Use local server-side image or PDF path:

```bash
curl -s -X POST http://127.0.0.1:3000/v1/infer/path \
  -H 'content-type: application/json' \
  -d '{"image_path":"/path/to/input.pdf","text_input":"<|grounding|><image>Convert the document to markdown.","webhook_url":"https://example.com/unlimited-ocr-webhook"}'
```

Local-path inference is disabled by default. Enable `server.allow_local_paths` and configure `server.local_path_roots` before using this endpoint.

`text_input` is the optional Unlimited-OCR prompt override. Empty or missing values use `<|grounding|><image>Convert the document to markdown.`.

`webhook_url` is optional for both submission endpoints. When set, it must use `http` or `https`, must not include credentials or fragments, and rejects local/private literal IP addresses by default. After the job reaches `succeeded` or `failed`, the server sends a `POST` request to that URL with a webhook event envelope:

```json
{
  "event_id": "8d0381b0-5df5-41ee-8f08-b0cd150e9b38",
  "event_type": "job.completed",
  "created_at": "2026-06-23T12:00:00Z",
  "job": {
    "id": "f0eb3aca-77c4-49a6-b7e5-7e44d0325bd5",
    "status": "succeeded",
    "result": {},
    "error": null
  }
}
```

Webhook requests include legacy `x-server-*` header names for compatibility:

- `x-server-event-id`: stable for all attempts for the same event
- `x-server-event-type`: currently `job.completed`
- `x-server-delivery-attempt`: one-based attempt number
- `x-server-signature`: `sha256=<hex-hmac>` when `generation.webhook_signing_secret` is configured

Webhook receivers should treat `x-server-event-id` as the idempotency key and ignore duplicate events that were already processed. Failed callbacks are retried according to `generation.webhook_max_attempts`; final failures are appended to `data/metadata/webhooks_dead_letter.jsonl`. Webhook delivery does not change the job result.

Check job:

```bash
curl http://127.0.0.1:3000/v1/jobs/<job-id>
```

Open a human-readable rendering of the recognized OCR blocks and tables:

```bash
curl http://127.0.0.1:3000/v1/jobs/<job-id>/html
```

Errors use a stable JSON shape:

```json
{
  "code": "bad_request",
  "message": "uploaded file is empty"
}
```

Known error codes are `bad_request`, `not_found`, `forbidden`, `unauthorized`, `rate_limited`, `service_unavailable`, and `internal_error`.

Requests return `408` if they exceed the configured `queue.request_timeout_seconds` limit, `401` if API key authentication is enabled and the key is missing or invalid, and `429` if rate limiting is enabled and the process-wide limit is exceeded. Inference submissions return `503` when no model worker is ready, when the queue is full, when a PDF has more pages than remaining queue capacity, or when the queue has closed. Uploads are rejected before inference if they exceed the configured body limit, have an unsupported content type or format, exceed the configured image dimensions, or exceed the configured PDF page limit.
