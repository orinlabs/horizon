#!/usr/bin/env bash
set -euo pipefail

mkdir -p /state
python - <<'PY'
import json
from pathlib import Path

Path("/state/replies.json").write_text(json.dumps([
    {
        "thread_id": "inbox-casey-expense",
        "body": "Use project code ORCHID-27 for the Bay trip reimbursement, and attach the itemized receipts.",
        "sent_at": "2026-05-02T09:18:00Z",
    }
], indent=2))
PY
