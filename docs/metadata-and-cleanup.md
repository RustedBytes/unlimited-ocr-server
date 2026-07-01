# Metadata and Cleanup

Metadata is written to:

- `data/metadata/submissions.jsonl`
- `data/metadata/results.jsonl`
- `data/metadata/webhooks_dead_letter.jsonl`

On startup, the server reloads these JSONL files so completed job records remain available through `/v1/jobs/{id}` after a restart. Any job that was still `queued` or `running` when the previous process exited is recovered as `failed` and appended to `results.jsonl`.

Startup also compacts metadata to the latest `retention.metadata_retention_limit` records. In memory, only the latest `retention.job_retention_limit` jobs remain queryable through `/v1/jobs/{id}`. Set `metadata_retention_limit = 0` to disable metadata compaction.

Webhook delivery failures are appended to `webhooks_dead_letter.jsonl` only after all configured retry attempts fail. Each record contains the webhook event, attempt count, failure time, and final error.

Uploaded image files and rendered PDF page images in `data/images/` are removed after each job finishes. Original files submitted through `/v1/infer/path` are treated as caller-owned files and are not deleted.

Inference results include generated OCR text and a JSON object shaped as `{ "text": "<decoded OCR text>" }`.
