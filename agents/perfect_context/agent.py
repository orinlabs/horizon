"""PerfectContextAgent — retrieval upper-bound baseline.

This agent simulates a *perfect memory harness*: instead of being told the
answer in prose, it is handed the exact verbatim slice(s) of its own
prior-session trace that contain the fact(s) needed for the task. Those
snippets are loaded directly into the context window as "recalled history".

It is given NO grep/shell tool and NO solvability hint file — only the
case's normal task tools (the eval-author-blessed CLI tools surfaced through
the sandbox `tools.json`). It must use those tools to explore the *current*
environment (find the triggering request, the real thread/recipient ids,
etc.) and then complete the task, grounding any prior-session detail in the
recalled snippets.

Where the snippets come from
----------------------------
`evals/<slug>/trace_pointer.json` (authored by `scripts/find_trace_pointers.py`)
records the 1-indexed line range(s) in `/workdir/trace.jsonl` that hold the key
fact — line numbers ONLY, never the answer text. At runtime the agent reads
those line ranges and `awk`s exactly those lines out of the sandbox trace.
Because the pointer carries no prose answer, a pass here genuinely requires the
fact to be present in (and recovered from) the trace.

Contrast with the other baselines:
  - `tools_only`     : never reads the trace (memoryless floor).
  - `trace_rag`      : must *discover* the right chunks via embedding search.
  - `perfect_context`: is handed exactly the right history slice — the ceiling.
    A failure here is a reasoning/acting failure, not a retrieval failure.

Run it with::

    source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/108-24-no-recording-sessions-policy-v0 \\
        --agent-import-path perfect_context.agent:PerfectContextAgent \\
        -m anthropic/claude-sonnet-4.5 \\
        --ae OPENROUTER_API_KEY=$OPENROUTER_API_KEY \\
        --ae OPENROUTER_MANAGEMENT_KEY=$OPENROUTER_MANAGEMENT_KEY \\
        --ve OPENROUTER_API_KEY=$OPENROUTER_API_KEY
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_utils import (
    HorizonToolRegistry,
    load_environment_tools,
    read_trace_file,
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


TRACE_PATH = "/workdir/trace.jsonl"
CASE_TOML_PATH = "/case.toml"
# evals/<slug>/trace_pointer.json on the host (agent runs under repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
HOST_POINTER_BASENAME = "trace_pointer.json"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_STEPS = 24
ATIF_VERSION = "ATIF-v1.4"
MAX_EXEC_OUTPUT_CHARS = 12_000
MAX_SNIPPET_LINE_CHARS = 6_000
MAX_SNIPPET_TOTAL_CHARS = 30_000

SYSTEM_PROMPT = """You are an autonomous agent operating in a live task environment.

The user message contains VERBATIM excerpts recalled from your own prior-session
history with this user — the relevant pieces of memory for the current task.
Treat them as ground-truth recollections of what happened in earlier sessions.

You also have a set of task tools ({tool_names}) that act on the CURRENT
environment. Use them to:
  1. Explore current state to find what needs doing and the REAL target ids.
     The triggering request lives in current state, not in your memory, and its
     real thread_id is an opaque id (e.g. `th-01a8...`), never the
     human-readable name a description might use. List live threads
     (`sms_list_all`, `sms_list`, `email_list_threads`) and read the relevant
     one (`sms_read`, `email_read`) to get the exact thread_id / recipient.
  2. Complete the task by INVOKING the appropriate tool(s), grounding every
     prior-session detail in the recalled excerpts above. If current state
     contradicts your recalled history, trust the recalled history.

The task is graded SOLELY by the state your tool calls write (e.g.
/state/sms_sent.json) — your chat text is NEVER read by the grader. Composing a
reply in prose without invoking the tool scores ZERO. Do not end your turn
until you have executed the tool call(s) that complete the task.

Preserve discriminating details from the recalled excerpts exactly in the
graded output: names, pronouns, dates, times, timezones, email addresses, URLs,
policy exclusions, and one-off overrides. If the recalled history says a topic
or service is excluded/declined/forbidden, do not answer it substantively and do
not re-offer it — explicitly decline and cite the exclusion instead.
"""

NO_MEMORY_NOTE = (
    "(No prior-session memory snippets were available for this task. Use your "
    "tools to inspect current state and do your best.)"
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug_from_case_toml(case_toml_text: str) -> str | None:
    """Reconstruct the eval dir slug (e.g. ``108-24-no-recording-...-v0``) from
    the in-image ``/case.toml``: ``{entity}-{case_family}-{variant}``."""
    try:
        case = tomllib.loads(case_toml_text)
    except (tomllib.TOMLDecodeError, ValueError):
        return None
    base_slug = case.get("base_slug", "")
    meta = case.get("metadata", {})
    family = meta.get("case_family")
    variant = meta.get("variant")
    entity = None
    parts = base_slug.split("_")
    if len(parts) >= 2 and parts[0] == "entity" and parts[1].isdigit():
        entity = parts[1]
    if not (entity and family and variant):
        return None
    return f"{entity}-{family}-{variant}"


def _normalize_ranges(raw: Any) -> list[list[int]]:
    out: list[list[int]] = []
    for r in raw or []:
        try:
            a, b = int(r[0]), int(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if a > b:
            a, b = b, a
        out.append([a, b])
    return out


async def _resolve_pointer(environment: BaseEnvironment) -> tuple[list[list[int]], str, str | None]:
    """Return ``(line_ranges, source, slug)`` from the host pointer file."""
    case_toml_text = await read_trace_file(environment, CASE_TOML_PATH)
    slug = _slug_from_case_toml(case_toml_text) if case_toml_text else None
    if slug:
        host_path = REPO_ROOT / "evals" / slug / HOST_POINTER_BASENAME
        try:
            data = json.loads(host_path.read_text(encoding="utf-8"))
            ranges = _normalize_ranges(data.get("line_ranges"))
            if ranges:
                return ranges, f"host:{host_path}", slug
        except (OSError, json.JSONDecodeError):
            pass
    return [], "none", slug


def _build_awk_cond(ranges: list[list[int]]) -> str:
    parts = []
    for a, b in ranges:
        parts.append(f"NR=={a}" if a == b else f"(NR>={a}&&NR<={b})")
    return "||".join(parts)


async def _exec_capture(
    environment: BaseEnvironment, command: str, *, timeout_sec: int, head_cap: int
) -> str:
    """Exec a command and return stdout capped from the FRONT (head)."""
    last_error = ""
    for attempt, backoff_sec in enumerate((0.0, 1.0, 2.0)):
        if backoff_sec:
            await asyncio.sleep(backoff_sec)
        try:
            result = await environment.exec(command, timeout_sec=timeout_sec)
            return (result.stdout or "")[:head_cap]
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return f"(failed to load snippets: {last_error})"


async def _load_snippets(environment: BaseEnvironment, ranges: list[list[int]]) -> str:
    """awk the exact pointer lines out of the sandbox trace, prefixed with their
    1-indexed line number, each line truncated to bound context size."""
    cond = _build_awk_cond(ranges)
    cmd = (
        f"awk '{cond}{{print NR\": \"substr($0,1,{MAX_SNIPPET_LINE_CHARS})}}' "
        f"{TRACE_PATH}"
    )
    text = await _exec_capture(
        environment, cmd, timeout_sec=60, head_cap=MAX_SNIPPET_TOTAL_CHARS
    )
    return text.strip()


async def _exec_with_retries(
    environment: BaseEnvironment, command: str, *, timeout_sec: int
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


class PerfectContextAgent(BaseAgent):
    """Retrieval ceiling: exact prior-session snippets in context + task tools."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "perfect-context"

    def version(self) -> str | None:
        return "0.2.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "PerfectContextAgent requires OPENROUTER_API_KEY in the host env."
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from openai import AsyncOpenAI

        management_key = os.environ["OPENROUTER_MANAGEMENT_KEY"]
        model = self.model_name or DEFAULT_MODEL

        t_start = time.monotonic()
        trial_label = f"horizon-perfect-context-{uuid.uuid4().hex[:8]}"

        steps: list[Step] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        chat_cost_usd = 0.0
        t_ingest_done = t_start
        t_end = t_start

        async with trial_subkey(
            management_key=management_key, label=trial_label
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=tk.key
            )

            # Task tools the agent is allowed to call (no shell, no trace tool).
            tool_registry = await load_environment_tools(environment)

            # Resolve the per-case pointer (line numbers only) and pull exactly
            # those lines out of the trace as verbatim recalled history.
            line_ranges, pointer_source, slug = await _resolve_pointer(environment)
            if line_ranges:
                snippets = await _load_snippets(environment, line_ranges)
            else:
                snippets = ""
            memory_block = snippets or NO_MEMORY_NOTE

            t_ingest_done = time.monotonic()

            system_prompt = SYSTEM_PROMPT.format(
                tool_names=", ".join(f"`{n}`" for n in tool_registry.names) or "(none)"
            )
            task_message = (
                "Recalled excerpts from your prior-session history "
                "(verbatim trace lines, prefixed with their line number):\n\n"
                f"{memory_block}\n\n"
                "----\n\n"
                f"Current task:\n\n{instruction}"
            )
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

            tools_schema = list(tool_registry.openrouter_tools)

            for turn_idx in range(MAX_STEPS):
                messages = [
                    {"role": "system", "content": system_prompt},
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
                    if name in tool_registry:
                        payload = await tool_registry.call(
                            environment,
                            name,
                            args,
                            output_char_cap=MAX_EXEC_OUTPUT_CHARS,
                        )
                    else:
                        payload = {
                            "exit_code": 127,
                            "stdout": "",
                            "stderr": (
                                f"unknown tool: {name}. Available: "
                                f"{', '.join(tool_registry.names) or '(none)'}."
                            ),
                        }

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
                "trace_path": TRACE_PATH,
                "slug": slug,
                "pointer_source": pointer_source,
                "pointer_found": pointer_source != "none",
                "line_ranges": line_ranges,
                "snippet_chars": len(snippets),
                "tools": list(tool_registry.names),
                "max_steps": MAX_STEPS,
                "timing_seconds": {
                    "ingest": round(t_ingest_done - t_start, 3),
                    "chat": round(t_end - t_ingest_done, 3),
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
