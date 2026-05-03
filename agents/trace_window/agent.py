"""TraceWindowAgent — sliding-window baseline.

Keeps only the last ``WINDOW_SIZE`` events of ``/workdir/trace.jsonl`` and
stuffs that compact slice into the user prompt. No retrieval, no
summarization, no memory store — a true floor for "what if the harness
just keeps recent context and discards everything else?". If the more
sophisticated agents (``trace_dump``, ``trace_rag``, ``hermes``) don't
beat this on a given eval, the extra ingestion isn't earning its keep.

Tool ownership matches the other harnesses: the environment publishes
``/tools/tools.json`` and this agent passes those schemas straight to the
LLM SDK.
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
WINDOW_SIZE = 100
MAX_STEPS = 10
DEFAULT_MODEL = "openai/gpt-4o-mini"
ATIF_VERSION = "ATIF-v1.4"

SYSTEM_PROMPT = (
    "You are an autonomous agent. You have a *limited* recent-history window "
    "from a prior session — older events have been discarded. Use the tools "
    "to complete the task. Stop when the task's success condition is met."
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


def _format_event(line: str) -> str:
    """Render one JSONL event as a compact single-line summary."""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line.strip()[:240]
    ts = obj.get("timestamp", "")
    msg = obj.get("message_data") or {}
    role = msg.get("role") or msg.get("type") or "?"
    if msg.get("type") == "function_call":
        name = msg.get("name", "?")
        args = msg.get("arguments", "")[:120]
        return f"[{ts}] tool_call {name}({args})"
    if msg.get("type") == "function_call_output":
        out = (msg.get("output") or "")[:200].replace("\n", " ")
        return f"[{ts}] tool_output {out}"
    if msg.get("type") == "message" or msg.get("role"):
        content = msg.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text") or p.get("content") or "")[:200] for p in content
            )
        content = (str(content or "")[:240]).replace("\n", " ")
        return f"[{ts}] {role}: {content}"
    return f"[{ts}] {json.dumps(msg)[:240]}"


class TraceWindowAgent(BaseAgent):
    """Sliding-window-over-events baseline agent."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-window"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceWindowAgent requires OPENROUTER_API_KEY in the host env."
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
        window_size = int(os.environ.get("TRACE_WINDOW_SIZE", WINDOW_SIZE))

        trace_result = await environment.exec(f"cat {TRACE_PATH} 2>/dev/null || true")
        raw = (trace_result.stdout or "").strip()
        all_lines = raw.splitlines() if raw else []
        recent = all_lines[-window_size:]
        formatted = "\n".join(_format_event(line) for line in recent)
        if not formatted:
            formatted = "(no trace available)"

        registry = await load_environment_tool_registry(environment)
        if registry is None:
            raise RuntimeError("TraceWindowAgent requires /tools/tools.json in the env.")
        tools = get_environment_tool_schemas(registry)

        user_message = (
            f"You have access to the LAST {len(recent)} events of a longer "
            f"prior session (the full trace had {len(all_lines)} events, "
            f"older ones are discarded):\n\n"
            f"{formatted}\n\n"
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

        total_prompt = total_completion = 0

        for _ in range(MAX_STEPS):
            resp = await _chat_completion_with_retries(
                client,
                model=model,
                messages=messages,
                tools=tools,
                temperature=0,
            )
            if resp.usage:
                total_prompt += resp.usage.prompt_tokens or 0
                total_completion += resp.usage.completion_tokens or 0
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
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_steps=len(steps),
            ),
            extra={
                "window_size": window_size,
                "events_in_window": len(recent),
                "events_total": len(all_lines),
                "environment_tools": list(registry.tools),
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_prompt
        context.n_output_tokens = total_completion
