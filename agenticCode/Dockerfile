# Dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    tar \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# goose
ENV PATH="/root/.local/bin:${PATH}"
RUN wget -qO- https://github.com/block/goose/releases/download/stable/download_cli.sh | CONFIGURE=false bash \
    && ln -s /root/.local/bin/goose /usr/local/bin/goose

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# goose configs
RUN mkdir -p /root/.config/goose
COPY config.yaml /root/.config/goose/config.yaml

RUN mkdir -p /root/.config/goose/custom_providers
COPY custom_skynetollama.json /root/.config/goose/custom_providers/custom_skynetollama.json

# COPY debug_tools_recipe.yaml .
COPY APItoken.py coordinator_recipe.yaml predictor_subagent_recipe.yaml evaluator_subagent_recipe.yaml mcp_server.py classic_runner.py compare.py llm_proxy.py metrics.py .


# env vars
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MLFLOW_EXPERIMENT_NAME=goose-traffic-detector
ENV MLFLOW_EXPERIMENT_ID=0
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://mlflow:5000
ENV OTEL_TRACES_EXPORTER=otlp
ENV OTEL_METRICS_EXPORTER=none
ENV OTEL_LOGS_EXPORTER=none

# entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]