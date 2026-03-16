# Multi-stage Dockerfile for healingstone reconstruction pipeline
# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system deps for geometry libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.lock ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .[runtime,pipeline2d]

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONPATH=/app/src
ENV MPLCONFIGDIR=/tmp/matplotlib_cache
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN useradd -m -r healingstone && \
    mkdir -p /app/artifacts /tmp/matplotlib_cache && \
    chown -R healingstone:healingstone /app /tmp/matplotlib_cache

USER healingstone

ENTRYPOINT ["python", "-m", "healingstone.pipeline.run_pipeline"]
CMD ["--help"]
