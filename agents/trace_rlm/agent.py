"""TraceRlmAgent — Recursive Language Model (RLM) baseline.

Faithful implementation of Recursive Language Models (Zhang, Kraska &
Khattab, 2026 — https://arxiv.org/abs/2512.24601) over the prior-session
trace.

The idea: rather than stuffing the (2–36M token) trace into the model's
context window, the trace is loaded as a **variable inside a persistent
Python REPL**. A *root* LM is given only the task + a description of the
REPL (and the trace's size), and writes code to ``peek``, ``grep``,
``partition + map``, and — crucially — launch **recursive sub-LM calls**
(depth=1) over slices of the trace via ``recurse(context, query)`` /
``llm(prompt)``. No single LM call ever has to hold the whole trace, which
sidesteps "context rot" and scales past the context window.

This is the recursive cousin of two existing baselines:
  - ``trace_shell_context`` greps the raw trace with shell but has no
    recursion and stuffs the whole file into context.
  - ``trace_window`` keeps only a recent slice.
RLM keeps the trace out of the root context entirely and lets the model
decide, at test time, how to decompose it.

Two tools are exposed to the root LM:
  - ``repl_exec(code)``: run Python in a persistent namespace pre-loaded
    with ``trace`` (full text), ``lines`` (raw JSONL lines), ``events``
    (parsed dicts), and the ``llm`` / ``recurse`` sub-call helpers.
  - ``shell_exec(command)``: act on the task environment.

Per the paper, root and recursive LMs differ: a capable root model drives
the REPL while a cheaper ``RLM_SUB_MODEL`` (default ``openai/gpt-5-mini``)
answers the recursive sub-queries. All calls route through the per-trial
OpenRouter sub-key so cost is fully attributed.

Run it with::

    source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/214-30-original-scope-recall-v0 \\
        --agent-import-path trace_rlm.agent:TraceRlmAgent \\
        -m openai/gpt-5
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from typing import Any

from agent_utils import read_trace_file, trial_subkey, usage_cost
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
# Root LM REPL/acting loop budget. RLM trajectories are longer than a flat
# tool loop (peek → grep → map → answer → act), so this is higher than the
# retrieval agents' MAX_STEPS.
MAX_STEPS = 30
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
# Cheaper recursive sub-LM (paper pairs GPT-5 root with GPT-5-mini sub).
DEFAULT_SUB_MODEL = "openai/gpt-5-mini"
ATIF_VERSION = "ATIF-v1.4"
MAX_EXEC_OUTPUT_CHARS = 12_000
# Truncation cap for a single REPL cell's captured stdout fed back to the root
# LM — keeps the root context from ballooning if it prints a huge slice.
MAX_REPL_OUTPUT_CHARS = 8_000
# Per-cell wall-clock budget. Recursive sub-calls inside a cell can be slow
# (each is a blocking LM call); this bounds a runaway loop without killing
# legitimate partition+map passes.
REPL_CELL_TIMEOUT_SEC = 600
# Per recursive/leaf LM call output cap and depth. Depth is fixed at 1 per the
# paper (root may call LMs, not other RLMs).
MAX_SUB_OUTPUT_CHARS = 6_000

SYSTEM_PROMPT = (
    "You are the ROOT language model of a Recursive Language Model (RLM).\n\n"
    "A prior agent session — possibly MILLIONS of tokens, far larger than your "
    "context window — is stored as a variable in a persistent Python REPL. You "
    "must NEVER try to read it all at once. Instead, write code to inspect and "
    "decompose it, and recurse over slices with sub-LM calls.\n\n"
    "Tools:\n"
    "  - `repl_exec(code)`: execute Python in a persistent notebook. Pre-loaded "
    "globals:\n"
    "      • `trace`  : str — the full prior session transcript.\n"
    "      • `lines`  : list[str] — `trace` split into raw JSONL event lines.\n"
    "      • `events` : list[dict] — `lines` parsed (skips unparseable lines).\n"
    "      • `llm(prompt, system=None) -> str` : one cheap sub-LM call.\n"
    "      • `recurse(context, query) -> str`  : ask the sub-LM `query` about a "
    "string `context` (your main recursion primitive — chunk the trace and map "
    "this over the chunks).\n"
    "    State persists across `repl_exec` calls (like Jupyter cells). `print(...)` "
    "to observe values; output is truncated.\n"
    "  - `shell_exec(command)`: run a shell command in the TASK ENVIRONMENT to "
    "act on the current world. Each function_call name in the trace is installed "
    "as a `/usr/local/bin` command of the same name with matching `--flag value` "
    "args.\n\n"
    "Suggested strategy:\n"
    "  1. PEEK: print `len(trace)`, `len(events)`, and a small slice to learn the "
    "structure.\n"
    "  2. NARROW: `grep` with substring/regex over `lines`/`events` to find "
    "relevant regions, OR partition the trace into chunks.\n"
    "  3. MAP/RECURSE: run `recurse(chunk, query)` over the relevant chunks and "
    "combine the answers.\n"
    "  4. ACT: use `shell_exec` to inspect the environment and complete the task, "
    "informed by what you recalled.\n\n"
    "Be economical: recursive calls cost money and time — narrow before you map. "
    "Complete the task and stop when its success condition is met."
)

REPL_EXEC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "repl_exec",
        "description": (
            "Execute Python in the persistent REPL holding the prior-session "
            "trace as a variable. Use print() to observe; call llm()/recurse() "
            "for sub-LM queries over slices."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."}
            },
            "required": ["code"],
            "additionalProperties": False,
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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


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


class _SubLMCaller:
    """Synchronous depth-1 sub-LM caller exposed inside the REPL.

    Runs on a worker thread (the REPL cell executes off the event loop), so a
    blocking ``openai.OpenAI`` client is the clean fit. Accumulates token/cost
    usage under a lock since a single cell may fan out many calls.
    """

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self._model = model
        self._lock = threading.Lock()
        self.calls = 0
        self.cost_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def llm(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": str(prompt)})
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0,
                extra_body={"usage": {"include": True}},
            )
        except Exception as exc:  # surfaced into the cell so the root LM adapts
            return f"[sub-LM error: {type(exc).__name__}: {exc}]"
        with self._lock:
            self.calls += 1
            self.cost_usd += usage_cost(resp)
            if resp.usage:
                self.prompt_tokens += resp.usage.prompt_tokens or 0
                self.completion_tokens += resp.usage.completion_tokens or 0
        content = resp.choices[0].message.content or ""
        return content[:MAX_SUB_OUTPUT_CHARS]

    def recurse(self, context: str, query: str) -> str:
        system = (
            "You are a recursive sub-call in a Recursive Language Model. Answer "
            "the QUERY using only the provided CONTEXT (a slice of a larger "
            "prior-session transcript). Be precise and concise; if the context "
            "does not contain the answer, say so explicitly."
        )
        prompt = f"CONTEXT:\n{context}\n\nQUERY:\n{query}"
        return self.llm(prompt, system=system)


class _Repl:
    """Persistent exec() namespace with per-cell stdout capture + timeout."""

    def __init__(self, namespace: dict[str, Any]) -> None:
        self._ns = namespace

    def run(self, code: str, timeout_sec: int) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def _target() -> None:
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(code, self._ns)  # noqa: S102 — RLM root LM drives this
                result["ok"] = True
            except Exception:
                result["ok"] = False
                buf.write("\n" + traceback.format_exc())
            result["output"] = buf.getvalue()

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout_sec)
        if thread.is_alive():
            # Can't safely kill the thread; report timeout and let it finish in
            # the background. The root LM should write a cheaper cell next.
            return {
                "ok": False,
                "output": (
                    f"[repl timeout after {timeout_sec}s — cell still running in "
                    f"background; write a smaller/cheaper cell]"
                ),
                "timeout": True,
            }
        output = result.get("output", "")
        if len(output) > MAX_REPL_OUTPUT_CHARS:
            head = output[: MAX_REPL_OUTPUT_CHARS // 2]
            tail = output[-MAX_REPL_OUTPUT_CHARS // 2 :]
            output = f"{head}\n...[{len(output)} chars total, truncated]...\n{tail}"
        return {"ok": result.get("ok", False), "output": output}


class TraceRlmAgent(BaseAgent):
    """Recursive Language Model baseline over the prior-session trace."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-rlm"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceRlmAgent requires OPENROUTER_API_KEY in the host env."
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from openai import AsyncOpenAI

        management_key = os.environ["OPENROUTER_MANAGEMENT_KEY"]
        chat_model = self.model_name or DEFAULT_CHAT_MODEL
        sub_model = os.environ.get("RLM_SUB_MODEL", DEFAULT_SUB_MODEL)

        t_start = time.monotonic()
        trial_label = f"horizon-trace-rlm-{uuid.uuid4().hex[:8]}"

        total_pt = total_ct = 0
        root_cost_usd = 0.0
        n_repl_cells = 0
        all_lines: list[str] = []
        steps: list[Step] = []
        t_load_done = t_start
        t_end = t_start
        sub_caller: _SubLMCaller | None = None

        async with trial_subkey(
            management_key=management_key,
            label=trial_label,
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=tk.key
            )

            trace_text = await read_trace_file(environment, TRACE_PATH)
            all_lines = trace_text.splitlines() if trace_text else []

            # Parse events once; the REPL exposes raw lines and parsed dicts.
            events: list[dict[str, Any]] = []
            for line in all_lines:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            # The trace lives in the host REPL, not the sandbox — remove it so
            # shell_exec can't bypass the RLM by cat-ing the file.
            await _exec_with_retries(environment, f"rm -f {TRACE_PATH}", timeout_sec=10)

            sub_caller = _SubLMCaller(api_key=tk.key, model=sub_model)
            repl_ns: dict[str, Any] = {
                "trace": trace_text or "",
                "lines": all_lines,
                "events": events,
                "llm": sub_caller.llm,
                "recurse": sub_caller.recurse,
                "json": json,
                "re": __import__("re"),
            }
            repl = _Repl(repl_ns)
            t_load_done = time.monotonic()

            all_tools = [REPL_EXEC_TOOL, SHELL_EXEC_TOOL]
            user_message = (
                f"The prior session is loaded in the REPL: `trace` is "
                f"{len(trace_text or '')} chars across {len(all_lines)} event "
                f"lines ({len(events)} parsed into `events`). It is too large to "
                f"read directly — inspect and recurse over it.\n\n"
                f"Current task:\n\n{instruction}"
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
                    model=chat_model,
                    messages=messages,
                    tools=all_tools,
                    temperature=0,
                    extra_body={"usage": {"include": True}},
                )
                if resp.usage:
                    total_pt += resp.usage.prompt_tokens or 0
                    total_ct += resp.usage.completion_tokens or 0
                root_cost_usd += usage_cost(resp)
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
                        args = {"code": tc.function.arguments or ""}
                    name = tc.function.name

                    if name == "repl_exec":
                        code = str(args.get("code") or "")
                        # exec() blocks; run off the event loop so concurrent
                        # trials in the same harbor run aren't starved.
                        cell = await asyncio.to_thread(
                            repl.run, code, REPL_CELL_TIMEOUT_SEC
                        )
                        n_repl_cells += 1
                        payload = {
                            "exit_code": 0 if cell.get("ok") else 1,
                            "stdout": cell.get("output", ""),
                            "stderr": "",
                        }
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
                        model_name=chat_model,
                        message=choice.content or "",
                        tool_calls=atif_tool_calls,
                        observation=Observation(results=observations),
                        metrics=step_metrics,
                    )
                )

            t_end = time.monotonic()

        sub_cost = sub_caller.cost_usd if sub_caller else 0.0
        sub_calls = sub_caller.calls if sub_caller else 0
        sub_pt = sub_caller.prompt_tokens if sub_caller else 0
        sub_ct = sub_caller.completion_tokens if sub_caller else 0
        # Recursive sub-LM tokens count toward the trial's token totals too.
        total_pt += sub_pt
        total_ct += sub_ct

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
                "events_total": len(all_lines),
                "trace_chars": len(trace_text or ""),
                "repl_cells": n_repl_cells,
                "sub_model": sub_model,
                "recursive_calls": sub_calls,
                "timing_seconds": {
                    "load": round(t_load_done - t_start, 3),
                    "solve": round(t_end - t_load_done, 3),
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": tk.cost_usd_dict(
                    direct_total=round(root_cost_usd + sub_cost, 6),
                    breakdown={
                        "root": round(root_cost_usd, 6),
                        "recursive": round(sub_cost, 6),
                    },
                ),
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_pt
        context.n_output_tokens = total_ct
