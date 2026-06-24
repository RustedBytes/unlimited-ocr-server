FROM docker.io/library/rust:1-trixie AS builder

ARG APP_NAME=unlimited-ocr-server

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        libssl-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY docs/openapi.json ./docs/openapi.json

RUN set -eux; \
    cargo build --release --locked; \
    mkdir -p /out/bin /out/lib; \
    cp "target/release/${APP_NAME}" /out/bin/; \
    find target/release target/release/deps \
        -maxdepth 1 \
        \( -type f -o -type l \) \
        \( -name '*.so' -o -name '*.so.*' \) \
        -exec cp -L '{}' /out/lib/ \;

FROM docker.io/library/debian:trixie-slim AS runtime

ARG APP_NAME=unlimited-ocr-server

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /app --create-home --shell /usr/sbin/nologin unlimitedocr

WORKDIR /app

COPY --from=builder /out/bin/${APP_NAME} /usr/local/bin/${APP_NAME}
COPY --from=builder /out/lib/ /usr/local/lib/
COPY config.example.toml /app/config.example.toml

RUN mkdir -p /app/data /app/Unlimited-OCR \
    && chown -R unlimitedocr:unlimitedocr /app

ENV BIND_ADDR=0.0.0.0:3000 \
    DATA_DIR=/app/data \
    LD_LIBRARY_PATH=/usr/local/lib \
    RUST_LOG=info,ort=warn

USER unlimitedocr

EXPOSE 3000
VOLUME ["/app/data", "/app/Unlimited-OCR"]
STOPSIGNAL SIGTERM

ENTRYPOINT ["unlimited-ocr-server"]
