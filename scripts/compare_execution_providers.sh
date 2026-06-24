#!/usr/bin/env bash

# IMAGE_PATH=docs/demo.png  MODEL_PATH=Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx DECODE_MODEL_PATH=Unlimited-OCR/onnx/unlimited_ocr_decode.onnx PROVIDER_LIST="cpu cuda,cpu" scripts/compare_execution_providers.sh

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  IMAGE_PATH=/path/to/image.jpg scripts/compare_execution_providers.sh

Optional environment:
  PROVIDER_LIST         Space-separated provider lists to test. Default: "cpu cuda,cpu"
  MODEL_PATH            Prefill/full ONNX path. Default: Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx
  DECODE_MODEL_PATH     Decode ONNX path. Defaults to sibling unlimited_ocr_decode.onnx when present.
  HOST_PORT             Host port for the temporary server. Default: 3001
  MAX_NEW_TOKENS        Generation limit. Default: 256
  MODEL_POOL_SIZE       Worker count. Default: 1
  TEXT_INPUT            Optional prompt override.
  READY_TIMEOUT_SECONDS Readiness timeout per provider. Default: 180
  JOB_TIMEOUT_SECONDS   Job polling timeout per provider. Default: 300
  RUST_LOG              Container log level. Default: info,ort=warn
USAGE
}

require_command() {
  local name="$1"

  if ! command -v "$name" >/dev/null 2>&1; then
    echo "missing required command: $name" >&2
    exit 1
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_command curl
require_command jq
require_command podman

if [ -z "${IMAGE_PATH:-}" ]; then
  usage >&2
  exit 2
fi

if [ ! -f "$IMAGE_PATH" ]; then
  echo "image file does not exist: $IMAGE_PATH" >&2
  exit 2
fi

: "${PROVIDER_LIST:=cpu cuda,cpu}"
: "${MODEL_PATH:=Unlimited-OCR/onnx/unlimited_ocr_prefill.onnx}"
: "${HOST_PORT:=3001}"
: "${MAX_NEW_TOKENS:=256}"
: "${MODEL_POOL_SIZE:=1}"
: "${READY_TIMEOUT_SECONDS:=180}"
: "${JOB_TIMEOUT_SECONDS:=300}"
: "${RUST_LOG:=info,ort=warn}"

if [ -z "${DECODE_MODEL_PATH:-}" ] && [ -f "Unlimited-OCR/onnx/unlimited_ocr_decode.onnx" ]; then
  DECODE_MODEL_PATH="Unlimited-OCR/onnx/unlimited_ocr_decode.onnx"
fi

base_url="http://127.0.0.1:${HOST_PORT}"
run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="target/provider-benchmarks/${run_id}"
mkdir -p "$log_dir"

if curl -fsS "${base_url}/health" >/dev/null 2>&1; then
  echo "port ${HOST_PORT} is already serving ${base_url}; set HOST_PORT to a free port" >&2
  exit 2
fi

server_pid=""
container_name=""

cleanup_server() {
  if [ -n "$container_name" ]; then
    podman rm -f "$container_name" >/dev/null 2>&1 || true
  fi

  if [ -n "$server_pid" ]; then
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi

  server_pid=""
  container_name=""
}

trap cleanup_server EXIT

wait_for_port_free() {
  local deadline=$((SECONDS + 30))

  while (( SECONDS < deadline )); do
    if ! curl -fsS "${base_url}/health" >/dev/null 2>&1; then
      return 0
    fi

    sleep 1
  done

  echo "port ${HOST_PORT} is still serving ${base_url} after cleanup" >&2
  return 1
}

wait_for_ready() {
  local log_file="$1"
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if curl -fsS "${base_url}/ready" >/dev/null 2>&1; then
      return 0
    fi

    if ! kill -0 "$server_pid" >/dev/null 2>&1; then
      echo "server exited before readiness; see ${log_file}" >&2
      return 1
    fi

    sleep 1
  done

  echo "server did not become ready within ${READY_TIMEOUT_SECONDS}s; see ${log_file}" >&2
  return 1
}

submit_job() {
  local submit_file="$1"
  local form=(-F "image=@${IMAGE_PATH}")

  if [ -n "${TEXT_INPUT:-}" ]; then
    form+=(-F "text_input=${TEXT_INPUT}")
  fi

  curl -fsS "${form[@]}" "${base_url}/v1/infer" > "$submit_file"
}

poll_job() {
  local job_id="$1"
  local result_file="$2"
  local deadline=$((SECONDS + JOB_TIMEOUT_SECONDS))
  local tmp_file="${result_file}.tmp"

  while (( SECONDS < deadline )); do
    curl -fsS "${base_url}/v1/jobs/${job_id}" > "$tmp_file"
    mv "$tmp_file" "$result_file"

    local status
    status="$(jq -r '.status' "$result_file")"
    case "$status" in
      succeeded|failed)
        return 0
        ;;
    esac

    sleep 1
  done

  echo "job ${job_id} did not finish within ${JOB_TIMEOUT_SECONDS}s" >&2
  return 1
}

extract_log_value() {
  local pattern="$1"
  local key="$2"
  local log_file="$3"

  grep "$pattern" "$log_file" 2>/dev/null \
    | tail -n 1 \
    | sed -n "s/.*${key}=\\([^ ]*\\).*/\\1/p" \
    || true
}

printf 'provider\tstatus\telapsed_ms\tgenerated_tokens\toverall_tok_s\tprefill_ms\tdecode_ms\tdecode_tok_s\tbackend\tresponse\tlog\n'

for providers in $PROVIDER_LIST; do
  cleanup_server
  wait_for_port_free

  safe_name="$(printf '%s' "$providers" | tr -c '[:alnum:]' '_')"
  container_name="unlimited-ocr-bench-${safe_name}-$$"
  log_file="${log_dir}/${safe_name}.log"
  submit_file="${log_dir}/${safe_name}.submit.json"
  result_file="${log_dir}/${safe_name}.result.json"

  CONTAINER_NAME="$container_name" \
    HOST_PORT="$HOST_PORT" \
    MODEL_PATH="$MODEL_PATH" \
    DECODE_MODEL_PATH="${DECODE_MODEL_PATH:-}" \
    EXECUTION_PROVIDERS="$providers" \
    MODEL_POOL_SIZE="$MODEL_POOL_SIZE" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    JOB_TIMEOUT_SECONDS="$JOB_TIMEOUT_SECONDS" \
    REQUEST_TIMEOUT_SECONDS="$JOB_TIMEOUT_SECONDS" \
    RUST_LOG="$RUST_LOG" \
    bash run_podman.sh > "$log_file" 2>&1 &
  server_pid=$!

  wait_for_ready "$log_file"
  submit_job "$submit_file"

  job_id="$(jq -r '.id // empty' "$submit_file")"
  if [ -z "$job_id" ]; then
    echo "submit response did not include a job id; see ${submit_file}" >&2
    exit 1
  fi

  poll_job "$job_id" "$result_file"

  status="$(jq -r '.status' "$result_file")"
  elapsed_ms="$(jq -r '.result.elapsed_ms // "n/a"' "$result_file")"
  generated_tokens="$(jq -r '.result.generated_tokens // "n/a"' "$result_file")"
  overall_tok_s="$(jq -r 'if (.result.elapsed_ms // 0) > 0 and (.result.generated_tokens // 0) > 0 then (.result.generated_tokens * 1000 / .result.elapsed_ms) else "n/a" end' "$result_file")"
  backend="$(jq -r '.result.backend // "n/a"' "$result_file")"
  prefill_ms="$(extract_log_value 'stage=prefill' 'elapsed_ms' "$log_file")"
  decode_ms="$(extract_log_value 'stage=decode' 'elapsed_ms' "$log_file")"
  decode_tok_s="$(extract_log_value 'stage=decode' 'tokens_per_second' "$log_file")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$providers" \
    "$status" \
    "$elapsed_ms" \
    "$generated_tokens" \
    "$overall_tok_s" \
    "${prefill_ms:-n/a}" \
    "${decode_ms:-n/a}" \
    "${decode_tok_s:-n/a}" \
    "$backend" \
    "$result_file" \
    "$log_file"

  cleanup_server
done

cleanup_server
