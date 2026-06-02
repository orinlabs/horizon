"""TraceMem0Agent — wrapper around mem0ai for memory-as-a-service baseline.

mem0 is a popular open-source memory layer. It LLM-extracts "memories" from
input messages, stores them in a vector DB, and retrieves them by similarity
at query time. Unlike ``trace_rag`` (which embeds raw chunks of trace
events) and ``trace_keyword`` (BM25 over the same chunks), mem0 owns the
extraction → storage → retrieval pipeline end-to-end and represents a
"production memory framework" baseline for the matrix.

Ingests the FULL trace, chunked by wake-sleep cycle. A cycle starts at a
"You are being woken up from sleep" message and ends at the next sleep()
function_call. Each cycle becomes one mem0 input message (compact narrative
of all events in that cycle), so mem0 extracts facts from coherent session
chunks rather than individual JSONL lines. A 35k-event trace ≈ 700 cycles
which fits comfortably in the 30-min agent timeout.

Storage is ephemeral ChromaDB on /tmp (recreated per trial run).

Environment integration matches ``trace_shell_context`` / ``trace_rag``:
the agent exposes a single ``shell_exec`` tool alongside ``memory_search``.
The eval's Dockerfile installs per-tool wrappers in ``/usr/local/bin``
(e.g. ``inbox_list``, ``reply_send``) via ``horizon-install-tools``, so
the model invokes them as plain shell commands — same names and flags it
can see in the prior trace via ``memory_search`` results.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
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
MAX_STEPS = 12
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
ATIF_VERSION = "ATIF-v1.4"
MEM0_USER_ID = "horizon-trace"
MEM0_BATCH_SIZE = 2          # cycles per mem0.add() call; >2 hits the 8191-token
                             # embedding limit on text-embedding-3-small (mem0
                             # embeds the concatenated batch as a search query)
MEM0_INGEST_CONCURRENCY = 4  # parallel mem0.add() calls; chromadb serializes
                             # writes internally so higher values yield diminishing
                             # returns and may starve other I/O
TOP_K_DEFAULT = 5
MAX_FORMATTED_EVENT_CHARS = 800
MAX_EXEC_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = (
    "You are an autonomous agent.\n\n"
    "You have tools:\n"
    "  - `memory_search`: query a mem0 memory store derived from the prior "
    "session. The store contains LLM-extracted facts; query in natural "
    "language. You have NO direct access to the raw trace.\n"
    "  - `shell_exec`: run a shell command in the task environment. The prior "
    "trace shows what shell commands are available (each function_call name "
    "is installed as a shell command of the same name in `/usr/local/bin` "
    "with matching `--flag value` arguments). Use this to act on the current "
    "world.\n\n"
    "Workflow: ALWAYS start with `shell_exec` to inspect the current environment "
    "(e.g. `task_list`, `sms_list`, `show_account` — names match the trace's "
    "function_call events). The task is to act on the present situation — "
    "`memory_search` is for INFORMING that action with prior-session memory, "
    "not a destination in itself. Don't query memory repeatedly with rephrased "
    "versions of 'what should I do'; find the pending work in the environment "
    "and complete it.\n\n"
    "Complete the task and stop when its success condition is met."
)

MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Semantic search over mem0-extracted memories from the prior session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {
                    "type": "integer",
                    "description": f"Memories to return (default {TOP_K_DEFAULT}).",
                    "minimum": 1,
                    "maximum": 20,
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


def _format_event(line: str) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return line.strip()[:MAX_FORMATTED_EVENT_CHARS]
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
    return f"[{ts}] {role}: {(str(content or ''))[:MAX_FORMATTED_EVENT_CHARS]}"


def _is_wake(obj: dict[str, Any]) -> bool:
    """A 'wake' marker = a system/user message containing the standard wake phrase."""
    md = obj.get("message_data") or {}
    if md.get("type") in ("function_call", "function_call_output"):
        return False
    content = md.get("content")
    if isinstance(content, list):
        content = " ".join(
            str(p.get("text") or p.get("content") or "") for p in content
            if isinstance(p, dict)
        )
    return "being woken up from sleep" in str(content or "").lower()


def _is_sleep(obj: dict[str, Any]) -> bool:
    md = obj.get("message_data") or {}
    return md.get("type") == "function_call" and md.get("name") == "sleep"


def _chunk_by_wake_cycle(all_lines: list[str]) -> list[list[str]]:
    """Group JSONL event lines into wake-sleep cycles.

    A cycle starts at a wake message and ends at the next sleep() call.
    Events before the first wake are emitted as a "preamble" cycle so we
    don't drop trace prelude. Events after the final sleep without a
    subsequent wake are emitted as a "tail" cycle. mem0 ingests each cycle
    as a coherent narrative (one extract-facts pass per cycle), instead of
    treating every JSONL line as its own message.
    """
    cycles: list[list[str]] = []
    current: list[str] = []
    seen_first_wake = False
    for line in all_lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            current.append(line)
            continue
        if _is_wake(obj):
            if current:
                cycles.append(current)
            current = [line]
            seen_first_wake = True
        elif _is_sleep(obj):
            current.append(line)
            cycles.append(current)
            current = []
        else:
            current.append(line)
    if current:
        cycles.append(current)
    # Drop trivial cycles: empties (back-to-back boundaries) and singletons
    # that are usually just the empty function_call_output of a sleep() call
    # — no facts there for mem0 to extract.
    return [c for c in cycles if len(c) >= 2]


def _build_mem0(chroma_path: str, api_key: str) -> Any:
    """Construct a mem0 Memory pointed at OpenRouter for LLM/embeddings."""
    from mem0 import Memory

    extraction_model = os.environ.get("MEM0_EXTRACTION_MODEL", DEFAULT_EXTRACTION_MODEL)
    embedding_model = os.environ.get("MEM0_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": extraction_model,
                "openai_base_url": "https://openrouter.ai/api/v1",
                "api_key": api_key,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": embedding_model,
                "openai_base_url": "https://openrouter.ai/api/v1",
                "api_key": api_key,
                "embedding_dims": 1536,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "horizon_trace",
                "path": chroma_path,
            },
        },
    }
    return Memory.from_config(config)


class TraceMem0Agent(BaseAgent):
    """mem0-backed memory baseline."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-mem0"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceMem0Agent requires OPENROUTER_API_KEY in the host env."
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

        t_start = time.monotonic()
        trial_label = f"horizon-trace-mem0-{uuid.uuid4().hex[:8]}"

        # Collected inside the `async with`; consumed after exit when
        total_pt = total_ct = 0
        chat_cost_usd = 0.0
        n_added = 0
        n_batches = 0
        ingest_seconds = 0.0
        all_lines: list[str] = []
        recent: list[str] = []
        searches_done: list[dict[str, Any]] = []
        steps: list[Step] = []
        t_ingest_done = t_start
        t_end = t_start

        async with trial_subkey(
            management_key=management_key,
            label=trial_label,
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=tk.key
            )

            # Use download_file (not exec/cat) so we don't truncate at the agent's
            # stdout cap — production traces can be tens of MB.
            trace_text = await read_trace_file(environment, TRACE_PATH)
            all_lines = trace_text.splitlines() if trace_text else []

            # Group by wake-sleep cycle so mem0 extracts facts from coherent
            # narratives instead of from individual JSONL lines.
            cycles = _chunk_by_wake_cycle(all_lines)
            # Render each cycle to one compact message (joined formatted events).
            cycle_messages = [
                "\n".join(_format_event(line) for line in cycle) for cycle in cycles
            ]
            recent = cycle_messages  # kept variable name for trajectory logging

            # Best-effort delete to force the model through memory_search.
            await _exec_with_retries(environment, f"rm -f {TRACE_PATH}", timeout_sec=10)

            chroma_path = tempfile.mkdtemp(prefix="trace_mem0_chroma_")
            try:
                memory = _build_mem0(chroma_path, tk.key)

                ingest_started = time.monotonic()
                # Parallel ingest: at any time, MEM0_INGEST_CONCURRENCY mem0.add
                # calls are in-flight. ChromaDB serializes writes internally so
                # we cap concurrency to avoid head-of-line blocking other I/O.
                sem = asyncio.Semaphore(MEM0_INGEST_CONCURRENCY)
                batches = [
                    recent[i : i + MEM0_BATCH_SIZE]
                    for i in range(0, len(recent), MEM0_BATCH_SIZE)
                ]

                async def _ingest_batch(batch_idx: int, batch: list[str]) -> int:
                    msgs = [{"role": "user", "content": c} for c in batch]
                    async with sem:
                        try:
                            result = await asyncio.to_thread(
                                memory.add, msgs, user_id=MEM0_USER_ID
                            )
                        except Exception as exc:
                            self.logger.warning(
                                "mem0 add batch %d failed: %s: %s",
                                batch_idx,
                                type(exc).__name__,
                                exc,
                            )
                            return 0
                    if isinstance(result, dict):
                        return len(result.get("results") or [])
                    return 0

                ingest_results = await asyncio.gather(
                    *[_ingest_batch(i, b) for i, b in enumerate(batches)]
                )
                n_batches = len(batches)
                n_added = sum(ingest_results)
                ingest_seconds = time.monotonic() - ingest_started
                self.logger.info(
                    "mem0 ingest: %d events -> %d batches -> %d memories in %.1fs",
                    len(recent),
                    n_batches,
                    n_added,
                    ingest_seconds,
                )

                t_ingest_done = time.monotonic()

                all_tools = [MEMORY_SEARCH_TOOL, SHELL_EXEC_TOOL]

                user_message = (
                    f"You have access to a mem0 memory store seeded from the most "
                    f"recent {len(recent)} events of a {len(all_lines)}-event prior "
                    f"session. {n_added} extracted memories are queryable via "
                    f"`memory_search`. The raw trace is no longer accessible.\n\n"
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

                        if name == "memory_search":
                            query = str(args.get("query") or "")
                            k = int(args.get("k") or TOP_K_DEFAULT)
                            try:
                                hits = await asyncio.to_thread(
                                    memory.search,
                                    query=query,
                                    user_id=MEM0_USER_ID,
                                    limit=k,
                                )
                            except Exception as exc:
                                payload = {
                                    "exit_code": 1,
                                    "stdout": "",
                                    "stderr": f"mem0 search failed: {exc!r}",
                                }
                            else:
                                results = hits.get("results", []) if isinstance(hits, dict) else hits
                                payload = {
                                    "exit_code": 0,
                                    "stdout": json.dumps(
                                        [
                                            {
                                                "memory": h.get("memory") if isinstance(h, dict) else str(h),
                                                "score": h.get("score") if isinstance(h, dict) else None,
                                            }
                                            for h in (results or [])
                                        ]
                                    ),
                                    "stderr": "",
                                }
                                searches_done.append({"query": query, "k": k, "n_hits": len(results or [])})
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
            finally:
                shutil.rmtree(chroma_path, ignore_errors=True)


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
                "events_ingested": len(recent),
                "ingest_batches": n_batches,
                "memories_extracted": n_added,
                "ingest_seconds": round(ingest_seconds, 2),
                "searches": searches_done,
                "timing_seconds": {
                    "ingest": round(t_ingest_done - t_start, 3),
                    "chat": round(t_end - t_ingest_done, 3),
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": tk.cost_usd_dict(direct_total=chat_cost_usd),
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_pt
        context.n_output_tokens = total_ct
