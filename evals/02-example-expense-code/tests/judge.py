from __future__ import annotations

import json
import os
import re
from pathlib import Path


REPLIES_PATH = Path(os.environ.get("REPLIES_PATH", "/state/replies.json"))
REWARD_JSON_PATH = Path(os.environ.get("REWARD_JSON_PATH", "/logs/verifier/reward.json"))
TARGET_THREAD_ID = "inbox-casey-expense"

CODE_PATTERN = re.compile(r"\borchid[- ]?27\b", re.IGNORECASE)
RECEIPT_PATTERN = re.compile(r"\breceipts?\b|\bitemized\b", re.IGNORECASE)
DISTRACTOR_PATTERN = re.compile(r"\blunch[- ]?11\b", re.IGNORECASE)


def load_replies() -> list[dict[str, object]]:
    if not REPLIES_PATH.exists():
        return []
    try:
        payload = json.loads(REPLIES_PATH.read_text())
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def main() -> int:
    replies = load_replies()
    target = [r for r in replies if r.get("thread_id") == TARGET_THREAD_ID]
    body = "\n".join(str(r.get("body") or "") for r in target)
    non_target = [r for r in replies if r.get("thread_id") != TARGET_THREAD_ID]

    metrics = {
        "sent_reply": bool(replies),
        "sent_to_target_thread": bool(target),
        "sent_only_to_target_thread": bool(replies) and not non_target,
        "mentions_project_code": bool(CODE_PATTERN.search(body)),
        "mentions_receipt_requirement": bool(RECEIPT_PATTERN.search(body)),
        "mentions_distractor_code": bool(DISTRACTOR_PATTERN.search(body)),
    }
    reward = int(
        metrics["sent_to_target_thread"]
        and metrics["sent_only_to_target_thread"]
        and metrics["mentions_project_code"]
        and metrics["mentions_receipt_requirement"]
        and not metrics["mentions_distractor_code"]
    )
    REWARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON_PATH.write_text(json.dumps({"reward": reward, "metrics": metrics, "reply": body}, indent=2))
    return 0 if reward else 1


if __name__ == "__main__":
    raise SystemExit(main())
