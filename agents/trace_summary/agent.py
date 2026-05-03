"""TraceSummaryAgent — rolling LLM-summarization baseline.

Production "memory" pattern: chunk the prior trace into fixed-size buckets,
LLM-summarize each older bucket into prose, keep the most recent
``LIVE_TAIL_EVENTS`` events at full fidelity, and stuff
``[summaries…] + [recent raw events]`` into the user prompt. No retrieval
tool, no embeddings — just compression.

This compresses traces that would otherwise be too large to prompt directly,
without the embedding cost of ``trace_rag``. A useful comparator for
"is RAG worth it vs. plain summarization?"
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from typing import Any

from environment_tools import (
    call_environment_tool,
    get_environment_tool_schemas,
    load_environment_tool_registry,
)
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
# 27k-event traces are common; 1500 events/bucket caps summarization at
# ~20 sequential LLM calls (~30-60s) instead of ~150.
BUCKET_EVENTS = 1500
LIVE_TAIL_EVENTS = 30
SUMMARY_TARGET_CHARS = 1_500
SUMMARY_MODEL = "openai/gpt-4o-mini"
MAX_BUCKETS = 30
MAX_STEPS = 10
DEFAULT_MODEL = "openai/gpt-4o-mini"
ATIF_VERSION = "ATIF-v1.4"

SYSTEM_PROMPT = (
    "You are an autonomous agent. Older events from a prior session have "
    "been replaced with bucket summaries; recent events are shown verbatim. "
    "Use the available tools to complete the task and stop when its success "
    "condition is met."
)

SUMMARY_INSTR = (
    "Summarize this slice of an autonomous agent's prior session in "
    f"≤{SUMMARY_TARGET_CHARS} characters. Preserve concrete facts that may "
    "be referenced later: vendor names, dollar amounts, dates, specific "
    "decisions, identifiers (thread ids, doc ids), and any unresolved tasks. "
    "Skip generic chit-chat and tool boilerplate."
)


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


def _format_event_for_summary(line: str) -> str:
    """Render one JSONL event into a readable line for the summarizer."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line.strip()[:400]
    ts = obj.get("timestamp", "")
    msg = obj.get("message_data") or {}
    if msg.get("type") == "function_call":
        return f"[{ts}] CALL {msg.get('name','?')} args={msg.get('arguments','')[:300]}"
    if msg.get("type") == "function_call_output":
        return f"[{ts}] OUT  {(msg.get('output') or '')[:400]}"
    role = msg.get("role") or msg.get("type") or "?"
    content = msg.get("content")
    if isinstance(content, list):
        content = " ".join(
            str(p.get("text") or p.get("content") or "")[:300] for p in content
        )
    return f"[{ts}] {role}: {(str(content or ''))[:400]}"


def _bucketize(lines: list[str], bucket_size: int) -> list[list[str]]:
    return [lines[i : i + bucket_size] for i in range(0, len(lines), bucket_size)]


async def _summarize_bucket(
    client: Any, bucket: list[str], summary_model: str
) -> tuple[str, int, int]:
    body = "\n".join(_format_event_for_summary(line) for line in bucket)
    resp = await _chat_completion_with_retries(
        client,
        model=summary_model,
        messages=[
            {"role": "system", "content": SUMMARY_INSTR},
            {"role": "user", "content": body},
        ],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip() or "(empty summary)"
    pt = (resp.usage.prompt_tokens if resp.usage else 0) or 0
    ct = (resp.usage.completion_tokens if resp.usage else 0) or 0
    return text, pt, ct


class TraceSummaryAgent(BaseAgent):
    """Rolling-summary baseline."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-summary"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceSummaryAgent requires OPENROUTER_API_KEY in the host env."
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
        chat_model = self.model_name or DEFAULT_MODEL
        summary_model = os.environ.get("TRACE_SUMMARY_MODEL", SUMMARY_MODEL)

        trace_result = await environment.exec(f"cat {TRACE_PATH} 2>/dev/null || true")
        raw = (trace_result.stdout or "").strip()
        all_lines = raw.splitlines() if raw else []
        n_total = len(all_lines)

        live_tail = all_lines[-LIVE_TAIL_EVENTS:]
        head = all_lines[: max(0, n_total - LIVE_TAIL_EVENTS)]
        buckets = _bucketize(head, BUCKET_EVENTS)
        # If a trace is so long that we'd still hit too many buckets,
        # widen each bucket geometrically until we're under MAX_BUCKETS.
        bucket_size = BUCKET_EVENTS
        while len(buckets) > MAX_BUCKETS:
            bucket_size *= 2
            buckets = _bucketize(head, bucket_size)

        summary_pt = summary_ct = 0
        summaries: list[str] = []
        for idx, bucket in enumerate(buckets):
            text, pt, ct = await _summarize_bucket(client, bucket, summary_model)
            summary_pt += pt
            summary_ct += ct
            summaries.append(f"## Bucket {idx + 1}/{len(buckets)} ({len(bucket)} events)\n{text}")

        live_block = "\n".join(_format_event_for_summary(line) for line in live_tail)
        summaries_block = "\n\n".join(summaries) if summaries else "(no older history)"

        registry = await load_environment_tool_registry(environment)
        if registry is None:
            raise RuntimeError("TraceSummaryAgent requires /tools/tools.json in the env.")
        tools = get_environment_tool_schemas(registry)

        user_message = (
            f"# Compressed prior session ({n_total} events total)\n\n"
            f"## Summaries of older buckets\n{summaries_block}\n\n"
            f"## Recent {len(live_tail)} events (verbatim)\n{live_block}\n\n"
            f"---\n\nCurrent task:\n\n{instruction}"
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

        chat_pt = chat_ct = 0

        for _ in range(MAX_STEPS):
            resp = await _chat_completion_with_retries(
                client,
                model=chat_model,
                messages=messages,
                tools=tools,
                temperature=0,
            )
            if resp.usage:
                chat_pt += resp.usage.prompt_tokens or 0
                chat_ct += resp.usage.completion_tokens or 0
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
                        model_name=chat_model,
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

                payload = await call_environment_tool(
                    environment,
                    registry,
                    name,
                    args,
                )

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
                    model_name=chat_model,
                    message=choice.content or "",
                    tool_calls=atif_tool_calls,
                    observation=Observation(results=observations),
                    metrics=step_metrics,
                )
            )

        total_pt = chat_pt + summary_pt
        total_ct = chat_ct + summary_ct

        trajectory = Trajectory(
            schema_version=ATIF_VERSION,
            session_id=str(uuid.uuid4()),
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=chat_model,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_pt,
                total_completion_tokens=total_ct,
                total_steps=len(steps),
            ),
            extra={
                "events_total": n_total,
                "buckets": len(buckets),
                "bucket_size": bucket_size,
                "live_tail_events": len(live_tail),
                "summary_model": summary_model,
                "summary_prompt_tokens": summary_pt,
                "summary_completion_tokens": summary_ct,
                "environment_tools": list(registry.tools),
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_pt
        context.n_output_tokens = total_ct
