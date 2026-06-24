# podman build -f Containerfile.gpu -t unlimited-ocr-server:gpu-local .

# podman volume create unlimited-ocr-data

# Each worker loads its own CUDA model session. Keep the local GPU script
# conservative by default; override with MODEL_POOL_SIZE=2 if memory allows.
: "${MODEL_POOL_SIZE:=1}"
: "${HOST_PORT:=3000}"

env_args=(
  -e MODEL_POOL_SIZE="$MODEL_POOL_SIZE"
)

add_env_if_set() {
  local name="$1"

  if [ -n "${!name:-}" ]; then
    env_args+=(-e "$name=${!name}")
  fi
}

add_env_if_set MODEL_PATH
add_env_if_set DECODE_MODEL_PATH
add_env_if_set CONFIG_PATH
add_env_if_set EXECUTION_PROVIDERS
add_env_if_set MAX_NEW_TOKENS
add_env_if_set RUST_LOG
add_env_if_set JOB_TIMEOUT_SECONDS
add_env_if_set REQUEST_TIMEOUT_SECONDS

name_args=()
if [ -n "${CONTAINER_NAME:-}" ]; then
  name_args=(--name "$CONTAINER_NAME")
fi

podman run --rm \
  "${name_args[@]}" \
  --device /dev/nvidia0 \
  --device /dev/nvidiactl \
  --device /dev/nvidia-uvm \
  --device /dev/nvidia-uvm-tools \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.580.167.08:/usr/lib/x86_64-linux-gnu/libcuda.so.580.167.08:ro \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so.1:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libcuda.so:/usr/lib/x86_64-linux-gnu/libcuda.so:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.167.08:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.167.08:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.167.08:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.167.08:ro \
  -v /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1:ro \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/app:/usr/local/cuda/lib64 \
  "${env_args[@]}" \
  -p "$HOST_PORT:3000" \
  -v "$PWD/Unlimited-OCR:/app/Unlimited-OCR:ro" \
  -v unlimited-ocr-data:/app/data:U \
  unlimited-ocr-server:gpu-local
