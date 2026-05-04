"""TraceWindowAgent — sliding-window baseline.

Keeps only the last ``WINDOW_SIZE`` events of ``/workdir/trace.jsonl`` and
stuffs that compact slice into the user prompt. No retrieval, no
summarization, no memory store — a true floor for "what if the harness
just keeps recent context and discards everything else?". If the more
sophisticated agents (``trace_shell_context``, ``trace_rag``) don't beat
this on a given eval, the extra ingestion isn't earning its keep.

Environment integration matches ``trace_shell_context``: the agent exposes
a single ``shell_exec`` tool. The eval's Dockerfile installs per-tool
wrappers in ``/usr/local/bin`` (e.g. ``inbox_list``, ``reply_send``) via
``horizon-install-tools``, so the model invokes them as plain shell
commands — same names and flags it can see in the prior trace.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_utils import cost_note, read_trace_file, trial_subkey, usage_cost
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
MAX_EXEC_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = (
    "You are an autonomous agent. You have a *limited* recent-history window "
    "from a prior session — older events have been discarded. The prior trace "
    "shows what shell commands are available in this environment (each "
    "function_call name maps to a shell command of the same name installed in "
    "``/usr/local/bin`` with matching ``--flag value`` arguments). Use "
    "``shell_exec`` to run them. Stop when the task's success condition is met."
)

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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


async def _chat_completion_with_retries(client: Any, **kwargs: Any) -> Any:
    last_decode_error: json.JSONDecodeError | None = None
    for attempt in range(3):
        try:
            return await client.chat.completions.create(**kwargs)
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
        from openai import AsyncOpenAI

        api_key = os.environ["OPENROUTER_API_KEY"]
        provisioning_key = os.environ.get("OPENROUTER_PROVISIONING_KEY")
        model = self.model_name or DEFAULT_MODEL
        window_size = int(os.environ.get("TRACE_WINDOW_SIZE", WINDOW_SIZE))

        t_start = time.monotonic()
        trial_label = f"horizon-trace-window-{uuid.uuid4().hex[:8]}"

        all_lines: list[str] = []
        recent: list[str] = []
        steps: list[Step] = []
        total_prompt = total_completion = 0
        chat_cost_usd = 0.0
        t_ingest_done = t_start
        t_end = t_start

        async with trial_subkey(
            provisioning_key=provisioning_key,
            fallback_key=api_key,
            label=trial_label,
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=tk.key,
            )

            # Use download_file (not exec/cat) so we don't truncate at the agent's
            # stdout cap — production traces can be tens of MB.
            trace_text = await read_trace_file(environment, TRACE_PATH)
            all_lines = trace_text.splitlines() if trace_text else []
            recent = all_lines[-window_size:]
            formatted = "\n".join(_format_event(line) for line in recent)
            if not formatted:
                formatted = "(no trace available)"

            t_ingest_done = time.monotonic()

            tools = [SHELL_EXEC_TOOL]

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

            steps.append(
                Step(
                    step_id=1,
                    timestamp=_now_iso(),
                    source="user",
                    message=user_message,
                )
            )

            for _ in range(MAX_STEPS):
                resp = await _chat_completion_with_retries(
                    client,
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=0,
                    extra_body={"usage": {"include": True}},
                )
                if resp.usage:
                    total_prompt += resp.usage.prompt_tokens or 0
                    total_completion += resp.usage.completion_tokens or 0
                chat_cost_usd += usage_cost(resp)
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

                    if name != "shell_exec":
                        payload = {
                            "exit_code": 127,
                            "stdout": "",
                            "stderr": f"unknown tool: {name}",
                        }
                    else:
                        command = str(args.get("command") or "")
                        timeout_sec = int(args.get("timeout_sec") or 60)
                        payload = await _exec_with_retries(
                            environment,
                            command,
                            timeout_sec=timeout_sec,
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

            t_end = time.monotonic()

        # `tk.usage_usd` and `tk.mode` are now resolved.
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
                "timing_seconds": {
                    "ingest": round(t_ingest_done - t_start, 3),
                    "chat": round(t_end - t_ingest_done, 3),
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": {
                    "total": tk.usage_usd,
                    "chat_direct": round(chat_cost_usd, 6),
                    "mode": tk.mode,
                    "_note": cost_note(tk.mode),
                },
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_prompt
        context.n_output_tokens = total_completion
