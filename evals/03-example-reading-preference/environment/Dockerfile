FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/ /app/
COPY tools/ /tools/

RUN mkdir -p /workdir /state /logs/verifier /logs/agent
