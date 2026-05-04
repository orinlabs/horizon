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
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_STEPS = 12
ATIF_VERSION = "ATIF-v1.4"
MAX_EXEC_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = """You are an autonomous agent operating inside a benchmark environment.

You have exactly one tool:
- shell_exec: run a shell command in the environment.

Use shell commands to inspect the environment and complete the task. The full
prior-session trace is included in this context on every turn.
"""

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


class TraceShellContextAgent(BaseAgent):
    """Minimal full-trace baseline with one shell exec tool."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-shell-context"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceShellContextAgent requires OPENROUTER_API_KEY in the host env."
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

        t_start = time.monotonic()
        trial_label = f"horizon-trace-shell-context-{uuid.uuid4().hex[:8]}"

        steps: list[Step] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        chat_cost_usd = 0.0
        trace_text = ""
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
            if not trace_text:
                raise RuntimeError(f"failed to read trace at {TRACE_PATH}: file empty or missing")

            t_ingest_done = time.monotonic()

            task_message = f"Current task:\n\n{instruction}"
            trace_message = (
                "Full prior-session trace follows. Treat this as the source of truth "
                "for prior-session facts.\n\n"
                f"{trace_text}"
            )
            conversation: list[dict[str, Any]] = [{"role": "user", "content": task_message}]

            steps.append(
                Step(
                    step_id=1,
                    timestamp=_now_iso(),
                    source="user",
                    message=task_message,
                )
            )

            for _ in range(MAX_STEPS):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": trace_message},
                    *conversation,
                ]
                resp = await _chat_completion_with_retries(
                    client,
                    model=model,
                    messages=messages,
                    tools=[SHELL_EXEC_TOOL],
                    temperature=0,
                    extra_body={"usage": {"include": True}},
                )
                if resp.usage:
                    total_prompt_tokens += resp.usage.prompt_tokens or 0
                    total_completion_tokens += resp.usage.completion_tokens or 0
                chat_cost_usd += usage_cost(resp)
                step_metrics = Metrics(
                    prompt_tokens=(resp.usage.prompt_tokens if resp.usage else 0) or 0,
                    completion_tokens=(resp.usage.completion_tokens if resp.usage else 0)
                    or 0,
                )

                choice = resp.choices[0].message
                tool_calls = list(choice.tool_calls or [])
                conversation.append(
                    {
                        "role": "assistant",
                        "content": choice.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in tool_calls
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
                            message=choice.content or "(done)",
                            metrics=step_metrics,
                        )
                    )
                    break

                atif_tool_calls: list[ToolCall] = []
                observations: list[ObservationResult] = []
                for tool_call in tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {"command": tool_call.function.arguments or ""}

                    name = tool_call.function.name
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
                            tool_call_id=tool_call.id,
                            function_name=name,
                            arguments=args,
                        )
                    )
                    observations.append(
                        ObservationResult(
                            source_call_id=tool_call.id,
                            content=json.dumps(payload),
                        )
                    )
                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
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
                total_prompt_tokens=total_prompt_tokens,
                total_completion_tokens=total_completion_tokens,
                total_steps=len(steps),
            ),
            extra={
                "trace_path": TRACE_PATH,
                "trace_chars": len(trace_text),
                "max_steps": MAX_STEPS,
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
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_prompt_tokens
        context.n_output_tokens = total_completion_tokens
