"""TraceKeywordAgent — BM25 retrieval baseline.

Same shape as ``trace_rag``: chunk the prior trace by UTC day, expose a
``trace_search(query, k)`` tool, run a tool-calling chat loop. The only
difference is the index — BM25 over per-day token bags instead of
neural embeddings. Useful as a "is the embedding cost actually buying
anything?" comparator and as a dirt-cheap floor for retrieval-based
agents (no embedding API calls at all).

Environment integration matches ``trace_shell_context`` / ``trace_rag``:
the agent exposes a single ``shell_exec`` tool alongside ``trace_search``.
The eval's Dockerfile installs per-tool wrappers in ``/usr/local/bin``
(e.g. ``inbox_list``, ``reply_send``) via ``horizon-install-tools``, so
the model invokes them as plain shell commands — same names and flags it
can see in retrieved trace chunks.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import uuid
from collections import Counter, defaultdict
from datetime import UTC, datetime
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)


TRACE_PATH = "/workdir/trace.jsonl"
MAX_STEPS = 12
DEFAULT_MODEL = "openai/gpt-4o-mini"
ATIF_VERSION = "ATIF-v1.4"
TOP_K_DEFAULT = 3
MAX_FORMATTED_EVENT_CHARS = 1_200
MAX_CHUNK_CHARS = 8_000
MAX_EXEC_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = (
    "You are an autonomous agent.\n\n"
    "You have tools:\n"
    "  - `trace_search`: keyword-search (BM25) the prior-session trace, "
    "chunked by UTC day. You have NO other access to the trace — always "
    "call this tool before assuming any prior-session detail.\n"
    "  - `shell_exec`: run a shell command in the task environment. The prior "
    "trace shows what shell commands are available (each function_call name "
    "is installed as a shell command of the same name in `/usr/local/bin` "
    "with matching `--flag value` arguments). Use this to act on the current "
    "world.\n\n"
    "Complete the task and stop when its success condition is met."
)

TRACE_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "trace_search",
        "description": (
            "BM25 search over the prior-session trace (chunked per UTC day). "
            "Returns the top-k days whose events most lexically match the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "k": {
                    "type": "integer",
                    "description": f"Number of chunks to return (default {TOP_K_DEFAULT}).",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
}

SHELL_EXEC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "shell_exec",
        "description": "Run a shell command in the task environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute in the environment.",
                },
                "timeout_sec": {
                    "type": "integer",
                    "description": "Command timeout in seconds.",
                    "default": 60,
                    "minimum": 1,
                    "maximum": 300,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


async def _chat_completion_with_retries(client: Any, **kwargs: Any) -> Any:
    last_decode_error: json.JSONDecodeError | None = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(**kwargs)
        except json.JSONDecodeError as exc:
            last_decode_error = exc
            await asyncio.sleep(1 + attempt)
    assert last_decode_error is not None
    raise last_decode_error


async def _exec_with_retries(
    environment: BaseEnvironment,
    command: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    last_error = ""
    for attempt, backoff_sec in enumerate((0.0, 1.0, 2.0, 4.0)):
        if backoff_sec:
            await asyncio.sleep(backoff_sec)
        try:
            result = await environment.exec(command, timeout_sec=timeout_sec)
            return {
                "exit_code": result.return_code,
                "stdout": (result.stdout or "")[-MAX_EXEC_OUTPUT_CHARS:],
                "stderr": (result.stderr or "")[-MAX_EXEC_OUTPUT_CHARS:],
            }
        except Exception as exc:
            last_error = f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
    return {"exit_code": 1, "stdout": "", "stderr": last_error}


def _format_event(line: str) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line.strip()[:MAX_FORMATTED_EVENT_CHARS]
    ts = obj.get("timestamp", "")
    msg = obj.get("message_data") or {}
    if msg.get("type") == "function_call":
        return f"[{ts}] CALL {msg.get('name','?')} args={msg.get('arguments','')[:600]}"
    if msg.get("type") == "function_call_output":
        return f"[{ts}] OUT  {(msg.get('output') or '')[:800]}"
    role = msg.get("role") or msg.get("type") or "?"
    content = msg.get("content")
    if isinstance(content, list):
        content = " ".join(
            str(p.get("text") or p.get("content") or "")[:600] for p in content
        )
    return f"[{ts}] {role}: {(str(content or ''))[:MAX_FORMATTED_EVENT_CHARS]}"


def _utc_day(line: str) -> str | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    ts = obj.get("timestamp")
    if not ts or not isinstance(ts, str):
        return None
    return ts[:10]  # YYYY-MM-DD


class _BM25:
    """Tiny in-memory Okapi BM25 (k1=1.5, b=0.75)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.df: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.doc_lens = [len(d) for d in docs]
        self.avgdl = sum(self.doc_lens) / max(1, len(docs))
        for d in docs:
            for term in set(d):
                self.df[term] += 1
        n = len(docs)
        for term, freq in self.df.items():
            # Robertson-Sparck-Jones idf, clamped to ≥ 0.
            self.idf[term] = max(0.0, math.log((n - freq + 0.5) / (freq + 0.5) + 1.0))

    def score(self, query: list[str], doc_idx: int) -> float:
        d = self.docs[doc_idx]
        if not d:
            return 0.0
        dl = self.doc_lens[doc_idx]
        tf = Counter(d)
        s = 0.0
        for term in query:
            if term not in self.idf:
                continue
            f = tf.get(term, 0)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
            s += self.idf[term] * (f * (self.k1 + 1)) / max(1e-9, denom)
        return s

    def top_k(self, query: list[str], k: int) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self.docs))]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0]


def _chunk_by_day(lines: list[str]) -> list[tuple[str, str]]:
    """Group events by UTC day → list of ``(day_label, formatted_block)``."""
    by_day: dict[str, list[str]] = defaultdict(list)
    for line in lines:
        day = _utc_day(line) or "unknown"
        by_day[day].append(_format_event(line))

    chunks: list[tuple[str, str]] = []
    for day in sorted(by_day):
        block = "\n".join(by_day[day])
        # Soft-split very long days so a single mega-chunk doesn't dominate.
        if len(block) > MAX_CHUNK_CHARS:
            buf: list[str] = []
            buf_len = 0
            part = 1
            for ev in by_day[day]:
                if buf_len + len(ev) > MAX_CHUNK_CHARS and buf:
                    chunks.append((f"{day}#{part:03d}", "\n".join(buf)))
                    buf, buf_len, part = [], 0, part + 1
                buf.append(ev)
                buf_len += len(ev) + 1
            if buf:
                chunks.append((f"{day}#{part:03d}", "\n".join(buf)))
        else:
            chunks.append((f"{day}#001", block))
    return chunks


class TraceKeywordAgent(BaseAgent):
    """BM25-over-day-chunks retrieval baseline."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-keyword"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceKeywordAgent requires OPENROUTER_API_KEY in the host env."
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        model = self.model_name or DEFAULT_MODEL

        trace_payload = await _exec_with_retries(
            environment,
            f"cat {TRACE_PATH} 2>/dev/null || true",
            timeout_sec=30,
        )
        if trace_payload["exit_code"] != 0 and not trace_payload["stdout"]:
            raise RuntimeError(f"failed to read trace: {trace_payload['stderr']}")
        raw = str(trace_payload["stdout"] or "").strip()
        all_lines = raw.splitlines() if raw else []

        chunks = _chunk_by_day(all_lines)
        chunk_labels = [c[0] for c in chunks]
        chunk_bodies = [c[1] for c in chunks]
        chunk_tokens = [_tokenize(body) for body in chunk_bodies]

        bm25 = _BM25()
        bm25.fit(chunk_tokens)

        # Best-effort delete the file so the model can't shortcut around the
        # search tool — matches trace_rag's behavior.
        await _exec_with_retries(environment, f"rm -f {TRACE_PATH}", timeout_sec=10)

        all_tools = [TRACE_SEARCH_TOOL, SHELL_EXEC_TOOL]

        user_message = (
            f"You are working on a task. There is no prior-session trace in "
            f"your prompt — call `trace_search` to recall anything from the "
            f"prior session. The trace was chunked into {len(chunks)} per-day "
            f"chunks across {len({c.split('#')[0] for c in chunk_labels})} "
            f"unique days.\n\nCurrent task:\n\n{instruction}"
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        steps: list[Step] = [
            Step(
                step_id=1,
                timestamp=_now_iso(),
                source="user",
                message=user_message,
            )
        ]

        total_pt = total_ct = 0
        searches_done: list[dict[str, Any]] = []

        for _ in range(MAX_STEPS):
            resp = await _chat_completion_with_retries(
                client,
                model=model,
                messages=messages,
                tools=all_tools,
                temperature=0,
            )
            if resp.usage:
                total_pt += resp.usage.prompt_tokens or 0
                total_ct += resp.usage.completion_tokens or 0
            step_metrics = Metrics(
                prompt_tokens=(resp.usage.prompt_tokens if resp.usage else 0) or 0,
                completion_tokens=(resp.usage.completion_tokens if resp.usage else 0) or 0,
            )

            choice = resp.choices[0].message
            tool_calls = list(choice.tool_calls or [])

            messages.append(
                {
                    "role": "assistant",
                    "content": choice.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                    or None,
                }
            )

            if not tool_calls:
                steps.append(
                    Step(
                        step_id=len(steps) + 1,
                        timestamp=_now_iso(),
                        source="agent",
                        model_name=model,
                        message=(choice.content or "(done)"),
                        metrics=step_metrics,
                    )
                )
                break

            atif_tool_calls: list[ToolCall] = []
            observations: list[ObservationResult] = []
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"command": tc.function.arguments or ""}
                name = tc.function.name

                if name == "trace_search":
                    query = str(args.get("query") or "")
                    k = int(args.get("k") or TOP_K_DEFAULT)
                    hits = bm25.top_k(_tokenize(query), k)
                    payload = {
                        "exit_code": 0,
                        "stdout": json.dumps(
                            [
                                {
                                    "chunk_id": chunk_labels[i],
                                    "score": round(float(s), 4),
                                    "body": chunk_bodies[i][:6000],
                                }
                                for i, s in hits
                            ]
                        ),
                        "stderr": "",
                    }
                    searches_done.append({"query": query, "k": k, "n_hits": len(hits)})
                elif name == "shell_exec":
                    command = str(args.get("command") or "")
                    timeout_sec = int(args.get("timeout_sec") or 60)
                    payload = await _exec_with_retries(
                        environment,
                        command,
                        timeout_sec=timeout_sec,
                    )
                else:
                    payload = {
                        "exit_code": 127,
                        "stdout": "",
                        "stderr": f"unknown tool: {name}",
                    }

                atif_tool_calls.append(
                    ToolCall(
                        tool_call_id=tc.id,
                        function_name=name,
                        arguments=args,
                    )
                )
                observations.append(
                    ObservationResult(
                        source_call_id=tc.id,
                        content=json.dumps(payload),
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(payload),
                    }
                )

            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    timestamp=_now_iso(),
                    source="agent",
                    model_name=model,
                    message=choice.content or "",
                    tool_calls=atif_tool_calls,
                    observation=Observation(results=observations),
                    metrics=step_metrics,
                )
            )

        trajectory = Trajectory(
            schema_version=ATIF_VERSION,
            session_id=str(uuid.uuid4()),
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=model,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_pt,
                total_completion_tokens=total_ct,
                total_steps=len(steps),
            ),
            extra={
                "events_total": len(all_lines),
                "chunks": len(chunks),
                "searches": searches_done,
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_pt
        context.n_output_tokens = total_ct
