# podman build -f Containerfile.gpu -t unlimited-ocr-server:gpu-local .

# podman volume create unlimited-ocr-data

# Each worker loads its own CUDA model session. Keep the local GPU script
# conservative by default; override with MODEL_POOL_SIZE=2 if memory allows.
: "${MODEL_POOL_SIZE:=1}"

env_args=(
  -e MODEL_POOL_SIZE="$MODEL_POOL_SIZE"
)

if [ -n "${MODEL_PATH:-}" ]; then
  env_args+=(-e MODEL_PATH="$MODEL_PATH")
fi

if [ -n "${CONFIG_PATH:-}" ]; then
  env_args+=(-e CONFIG_PATH="$CONFIG_PATH")
fi

podman run --rm \
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
  -p 3000:3000 \
  -v "$PWD/Unlimited-OCR:/app/Unlimited-OCR:ro" \
  -v unlimited-ocr-data:/app/data:U \
  unlimited-ocr-server:gpu-local
