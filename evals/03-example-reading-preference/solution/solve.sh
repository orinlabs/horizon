#!/usr/bin/env bash
set -euo pipefail

mkdir -p /state
python - <<'PY'
import json
from pathlib import Path

Path("/state/replies.json").write_text(json.dumps([
    {
        "thread_id": "inbox-nia-reading",
        "body": "Prioritize Scaling Memory for Agents, and prep concise bullet notes.",
        "sent_at": "2026-05-03T12:50:00Z",
    }
], indent=2))
PY
