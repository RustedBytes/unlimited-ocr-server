# Running the Server

Start the server:

```bash
cargo run
```

Default bind address is `127.0.0.1:3000`.

Open the browser UI:

```bash
open http://127.0.0.1:3000/
```

The browser UI template is embedded into the binary.

## Podman

Build the container image:

```bash
podman build -f Containerfile -t unlimited-ocr-server .
```

Run it with the downloaded model files mounted at the default path:

```bash
podman volume create unlimited-ocr-data
podman run --rm \
  -p 3000:3000 \
  -v "$PWD/Unlimited-OCR:/app/Unlimited-OCR:ro" \
  -v unlimited-ocr-data:/app/data:U \
  unlimited-ocr-server
```

The image does not include ONNX model files. Mount the `Unlimited-OCR/` directory into `/app/Unlimited-OCR`.

To use a custom config file:

```bash
podman run --rm \
  -p 3000:3000 \
  -v "$PWD/Unlimited-OCR:/app/Unlimited-OCR:ro" \
  -v "$PWD/config.toml:/app/config.toml:ro" \
  -v unlimited-ocr-data:/app/data:U \
  unlimited-ocr-server
```

## Compose

Build and run with Compose:

```bash
podman compose -f compose.yml up --build
```

Docker Compose works with the same file:

```bash
docker compose -f compose.yml up --build
```

The Compose file mounts `./Unlimited-OCR` into the container read-only and stores runtime data in the `unlimited-ocr-data` named volume.
