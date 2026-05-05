FROM ghcr.io/orinlabs/horizon-1-environment:latest

COPY tools/ /.horizon/tools/
COPY trace.jsonl /workdir/trace.jsonl

RUN horizon-install-tools