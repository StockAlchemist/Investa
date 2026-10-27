FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ build-essential libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain for the calculation core (builder stage only; the runtime
# image receives just the compiled extension). The core links the system
# libsqlite3 (libsqlite3-dev above), the library Python's sqlite3 module
# uses, so the process holds one SQLite; the runtime image already has it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

COPY rust/investa_core /build/investa_core
RUN pip install --no-cache-dir --prefix=/install /build/investa_core

FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV PYTHONPATH=/app/src

COPY --from=builder /install /usr/local
COPY src /app/src

# Runtime data directory (mount persistent volume at /app/data in production)
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["python", "src/server/main.py"]
