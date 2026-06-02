"""ToolsOnlyAgent — a deliberately memoryless baseline.

This agent exposes ONLY the environment's stateful CLI tools (the
``horizon-tools`` registry at ``/.horizon/tools/tools.json``) to the model.
It has no ``shell_exec`` tool and never reads ``/workdir/trace.jsonl``, so it
*cannot* consult the prior-session history at all. It only sees the live state
returned by the tools it calls.

Its purpose is a validity probe: if a memory benchmark can be passed by an
agent that provably never touches the trace, the eval is not actually
measuring long-horizon recall (e.g. because the answer is leaked into seeded
state like a profile ``notes`` field).

Run it with::

    source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/857-65-ulysses-dog-as-recurring-distraction-v0 \\
        --agent-import-path tools_only.agent:ToolsOnlyAgent \\
        -m anthropic/claude-sonnet-4.5 \\
        --ve OPENROUTER_API_KEY=$OPENROUTER_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from contextlib import asynccontextmanager

from agent_utils import (
    TrialKeyState,
    load_environment_tools,
    trial_subkey,
    usage_cost,
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


DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_STEPS = 24
ATIF_VERSION = "ATIF-v1.4"

# Note: this prompt intentionally tells the model it has NO access to past
# history. It is the live assistant, working only from what the tools return.
# We do not mention any trace file, and we do not give it a shell, so it has
# no way to read prior-session data even if it wanted to.
SYSTEM_PROMPT = """You are an autonomous assistant operating live inside a \
benchmark environment.

You have NO memory of any past conversation or session, and NO access to any \
historical logs or transcripts. The ONLY information available to you is what \
the tools below return when you call them right now.

Before you take ANY action, exhaustively gather ALL POSSIBLE information from \
the environment tools. Do not answer or act until you have inspected \
everything available to you. Specifically:
  - Call every read/list tool the environment exposes (account, profiles, \
    inbox/SMS threads, emails, tasks, documents, etc.).
  - For every entity that can be opened or read individually, open it: read \
    EVERY profile in the account, EVERY SMS thread, EVERY email thread, and \
    EVERY document — not just the ones that look relevant. Follow up on every \
    ID a tool returns.
  - Re-check for anything you may have missed. Assume the detail you need may \
    be buried in a profile note, a document, or a thread you would otherwise \
    skip. Leave no tool output unread.

Only after you have collected the complete current state should you decide what \
to do and take whatever actions the situation calls for. Your success depends \
on the real actions you take via these tools — replies you send and state you \
change persist in the environment. Reasoning without acting accomplishes \
nothing.

When the situation is fully handled, stop and give a short final summary.
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@asynccontextmanager
async def _maybe_trial_subkey(*, label: str):
    """Mint a cost-capped sub-key when a management key is present.

    Falls back to using ``OPENROUTER_API_KEY`` directly when no
    ``OPENROUTER_MANAGEMENT_KEY`` is configured, so this probe agent can run
    without the sub-key minting infrastructure.
    """
    management_key = os.environ.get("OPENROUTER_MANAGEMENT_KEY")
    if management_key:
        async with trial_subkey(management_key=management_key, label=label) as tk:
            yield tk
    else:
        yield TrialKeyState(key=os.environ["OPENROUTER_API_KEY"])


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


class ToolsOnlyAgent(BaseAgent):
    """Memoryless baseline: only the environment CLI tools, never the trace."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "tools-only"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "ToolsOnlyAgent requires OPENROUTER_API_KEY in the host env."
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from openai import AsyncOpenAI

        model = self.model_name or DEFAULT_MODEL

        t_start = time.monotonic()
        trial_label = f"horizon-tools-only-{uuid.uuid4().hex[:8]}"

        steps: list[Step] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        chat_cost_usd = 0.0
        t_end = t_start

        # Load ONLY the structured CLI tools. We never read the trace file.
        registry = await load_environment_tools(environment)
        tools_schema = registry.openrouter_tools

        async with _maybe_trial_subkey(label=trial_label) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=tk.key,
            )

            task_message = f"Current task:\n\n{instruction}"
            conversation: list[dict[str, Any]] = [
                {"role": "user", "content": task_message}
            ]

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
                    *conversation,
                ]
                resp = await _chat_completion_with_retries(
                    client,
                    model=model,
                    messages=messages,
                    tools=tools_schema,
                    temperature=0,
                    extra_body={"usage": {"include": True}},
                )
                if resp.usage:
                    total_prompt_tokens += resp.usage.prompt_tokens or 0
                    total_completion_tokens += resp.usage.completion_tokens or 0
                chat_cost_usd += usage_cost(resp)
                step_metrics = Metrics(
                    prompt_tokens=(resp.usage.prompt_tokens if resp.usage else 0) or 0,
                    completion_tokens=(
                        resp.usage.completion_tokens if resp.usage else 0
                    )
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
                        args = {}

                    name = tool_call.function.name
                    payload = await registry.call(environment, name, args)

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
                "reads_trace": False,
                "tool_names": registry.names,
                "max_steps": MAX_STEPS,
                "timing_seconds": {
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": tk.cost_usd_dict(direct_total=chat_cost_usd),
            },
        )
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_prompt_tokens
        context.n_output_tokens = total_completion_tokens
