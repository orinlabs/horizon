FROM ghcr.io/orinlabs/horizon-1-environment:latest

ARG HORIZON_EVAL_SLUG=02-example-expense-code

COPY tools/ /.horizon/tools/

RUN horizon-download-trace "$HORIZON_EVAL_SLUG" && horizon-install-tools