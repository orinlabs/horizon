from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


STATE_DIR = Path("/state")
REPLIES_PATH = STATE_DIR / "replies.json"

THREADS = [
    {
        "thread_id": "inbox-jordan-party",
        "sender": "Jordan Lee",
        "unread": True,
        "messages": [
            {
                "from": "Jordan Lee",
                "sent_at": "2026-05-01T16:20:00Z",
                "body": "Can you remind me which caterer we picked for the June team dinner? Finance wants one detail to identify the right quote.",
            }
        ],
    },
    {
        "thread_id": "inbox-mateo-snacks",
        "sender": "Mateo Ruiz",
        "unread": False,
        "messages": [
            {
                "from": "Mateo Ruiz",
                "sent_at": "2026-05-01T10:05:00Z",
                "body": "North Pier Pantry is handling Wednesday snacks. Different event from the team dinner.",
            }
        ],
    },
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_replies() -> list[dict[str, str]]:
    if not REPLIES_PATH.exists():
        return []
    try:
        payload = json.loads(REPLIES_PATH.read_text())
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def save_replies(replies: list[dict[str, str]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    REPLIES_PATH.write_text(json.dumps(replies, indent=2))


def inbox_list(limit: int) -> None:
    shown = THREADS[: max(0, limit)]
    lines = [f"INBOX THREADS ({len(shown)} shown)"]
    for thread in shown:
        latest = thread["messages"][-1]
        unread = "unread" if thread["unread"] else "read"
        lines.extend(
            [
                "----------------------------------------",
                f"- {thread['sender']} ({unread})",
                f"  thread_id: {thread['thread_id']}",
                f"  Latest: \"{latest['body']}\"",
            ]
        )
    print("\n".join(lines))


def inbox_read(thread_id: str) -> None:
    thread = next((item for item in THREADS if item["thread_id"] == thread_id), None)
    if thread is None:
        print(json.dumps({"ok": False, "error": f"Unknown thread: {thread_id}"}))
        return
    lines = [f"THREAD: {thread['sender']} ({thread_id})"]
    for message in thread["messages"]:
        lines.extend(
            [
                "----------------------------------------",
                f"From: {message['from']}",
                f"At: {message['sent_at']}",
                message["body"],
            ]
        )
    print("\n".join(lines))


def reply_send(thread_id: str, body: str) -> None:
    if not any(item["thread_id"] == thread_id for item in THREADS):
        print(json.dumps({"ok": False, "error": f"Unknown thread: {thread_id}"}))
        return
    replies = load_replies()
    payload = {"thread_id": thread_id, "body": body, "sent_at": now_iso()}
    replies.append(payload)
    save_replies(replies)
    print(json.dumps({"ok": True, "reply": payload}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tool_name", choices=["inbox_list", "inbox_read", "reply_send"])
    parser.add_argument("--reasoning")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--thread-id")
    parser.add_argument("--body")
    args = parser.parse_args()

    if args.tool_name == "inbox_list":
        inbox_list(args.limit)
    elif args.tool_name == "inbox_read":
        if args.thread_id is None:
            parser.error("--thread-id is required for inbox_read")
        inbox_read(args.thread_id)
    elif args.tool_name == "reply_send":
        if args.thread_id is None or args.body is None:
            parser.error("--thread-id and --body are required for reply_send")
        reply_send(args.thread_id, args.body)


if __name__ == "__main__":
    main()
