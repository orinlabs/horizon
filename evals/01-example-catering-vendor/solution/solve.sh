#!/usr/bin/env bash
set -euo pipefail

mkdir -p /state
python - <<'PY'
import json
from pathlib import Path

Path("/state/replies.json").write_text(json.dumps([
    {
        "thread_id": "inbox-jordan-party",
        "body": "We picked Blue Spruce Catering. Dana Kim was the contact, and the quote was $38 per person.",
        "sent_at": "2026-05-01T16:25:00Z",
    }
], indent=2))
PY
