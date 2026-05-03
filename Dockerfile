# syntax=docker/dockerfile:1
FROM python:3.12-slim

LABEL org.opencontainers.image.title="Horizon-1 Environment Base"
LABEL org.opencontainers.image.description="Public base image for Horizon-1 eval environments."
LABEL org.opencontainers.image.source="https://github.com/orinlabs/horizon-1"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        tmux \
        asciinema \
    && rm -rf /var/lib/apt/lists/*

ENV HORIZON_TRACE_PATH=/workdir/trace.jsonl
ENV HORIZON_TOOLS_DIR=/.horizon/tools

WORKDIR /workdir

RUN mkdir -p /workdir /state /logs/verifier /logs/agent /.horizon/tools /tests

COPY scripts/horizon_download_trace.py /usr/local/bin/horizon-download-trace
COPY scripts/horizon_install_tools.py /usr/local/bin/horizon-install-tools
RUN chmod +x /usr/local/bin/horizon-download-trace /usr/local/bin/horizon-install-tools