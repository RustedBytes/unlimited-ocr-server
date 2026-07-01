set dotenv-load := true

default:
    @just --list

fmt:
    cargo fmt

fmt-check:
    cargo fmt -- --check

check:
    cargo check

test:
    cargo test

test-one filter:
    cargo test {{filter}}

clippy:
    cargo clippy --all-targets --all-features

ci: fmt-check clippy test

run:
    cargo run

build:
    cargo build

release:
    cargo build --release --locked

open:
    open http://127.0.0.1:3000/

container-build:
    podman build -f Containerfile -t unlimited-ocr-server .

container-build-gpu:
    podman build -f Containerfile.gpu -t unlimited-ocr-server:gpu .

podman-run:
    bash run_podman.sh

compose-up:
    podman compose -f compose.yml up --build

compose-down:
    podman compose -f compose.yml down

compare-providers image_path:
    IMAGE_PATH={{image_path}} scripts/compare_execution_providers.sh
