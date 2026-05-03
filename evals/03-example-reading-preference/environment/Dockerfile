FROM ghcr.io/orinlabs/horizon-1-environment:latest

ARG HORIZON_EVAL_SLUG=03-example-reading-preference

COPY tools/ /.horizon/tools/

RUN horizon-download-trace "$HORIZON_EVAL_SLUG" && horizon-install-tools
