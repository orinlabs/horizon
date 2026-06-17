#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path


DEFAULT_DATASET_BASE_URL = (
    "https://huggingface.co/datasets/orinlabs/horizon-example-traces/resolve/main"
)


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1]:
        print("usage: horizon-download-trace <eval-slug>", file=sys.stderr)
        return 2

    slug = sys.argv[1]
    base_url = os.environ.get("HORIZON_TRACE_BASE_URL", DEFAULT_DATASET_BASE_URL)
    trace_path = Path("/workdir/trace.jsonl")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/{slug}/trace.jsonl"

    request = urllib.request.Request(url)
    if token := os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(request) as response:
        trace_path.write_bytes(response.read())

    if trace_path.stat().st_size == 0:
        raise RuntimeError(f"downloaded empty trace from {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
