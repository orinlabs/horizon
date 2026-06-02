"""TraceRagAgent — RAG baseline over the prior-session trace.

Ingestion
---------
During ``run()``, the agent reads ``/workdir/trace.jsonl``, groups events
by UTC day, renders each day as a human-readable block, and embeds the
block via OpenRouter's ``/v1/embeddings`` endpoint (model slug
``openai/text-embedding-3-small``). Day blocks are split into bounded-size
chunks before embedding so long production traces do not exceed provider
gateway limits. The chunks and their L2-normalized
embeddings live in memory as a numpy array — no external vector DB.

Retrieval + acting
------------------
The LLM sees ``trace_search(query, k=3)`` for prior-session recall, plus
the **task-specific** tools declared in the sandbox's
``/.horizon/tools/tools.json`` registry (e.g. ``inbox_list``, ``sms_read``,
``reply_send``). Those tools are loaded through ``HorizonToolRegistry``
and exposed as proper OpenRouter function-call entries — the model
never sees a generic ``shell_exec`` escape hatch, so it can only do
exactly what the eval author whitelisted. Each tool call is dispatched
back to its matching ``/usr/local/bin/<tool>`` wrapper via
``registry.call(...)``, which renders the function-call args into the
equivalent ``<tool> --flag value`` shell command.

The system prompt tells the agent that nothing about the trace is in the
user message — it has to call ``trace_search`` to recall anything from the
prior session.

Emits an ATIF-compliant ``trajectory.json``. Prompt/completion tokens from
both chat and embeddings are accumulated; embedding usage lives on
``trajectory.extra`` since ATIF ``final_metrics`` is chat-oriented.

Run it with::

    source .env && export OPENROUTER_API_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/01-example-catering-vendor \\
        --agent-import-path trace_rag.agent:TraceRagAgent \\
        -m openai/gpt-4o-mini
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from agent_utils import (
    HorizonToolRegistry,
    load_environment_tools,
    read_trace_file,
    summarize_call_log,
    timed_call,
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
MAX_STEPS = 12
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
ATIF_VERSION = "ATIF-v1.4"
TOP_K_DEFAULT = 3
MAX_FORMATTED_EVENT_CHARS = 1_200
MAX_CHUNK_CHARS = 8_000
EMBEDDING_BATCH_SIZE = 8
# Max in-flight embedding requests per case during ingest (bounded queue).
EMBED_INGEST_CONCURRENCY = 100
MAX_EXEC_OUTPUT_CHARS = 12_000
# On-disk cache for ingest embeddings. The embedding matrix for a trace is a
# pure function of (chunk texts, embedding model) — the chat model is
# irrelevant — so sweeping many chat models over the same task should embed
# each trace exactly once. We key the cache on a hash of the chunk texts +
# embedding model so a cache hit is byte-for-byte the same input. Bump
# EMBED_CACHE_VERSION whenever chunking/formatting changes in a way that would
# invalidate previously-cached matrices.
EMBED_CACHE_VERSION = "v1"
DEFAULT_EMBED_CACHE_DIR = Path(".cache") / "trace_rag_embeddings"

SYSTEM_PROMPT_TEMPLATE = (
    "You are an autonomous agent.\n\n"
    "You have tools:\n"
    "  - `trace_search`: semantic-search the prior-session trace, which has "
    "been chunked by UTC day and embedded. You have NO other access to the "
    "trace — always call this tool before assuming any prior-session detail.\n"
    "  - Task tools: {tool_names}. Each one matches the shell command of the "
    "same name in the prior trace; use them to act on the current world. "
    "There is no generic shell escape hatch.\n\n"
    "Workflow: ALWAYS start by inspecting the current environment (call task "
    "tools like `task_list`, `sms_list`, `show_account` etc. to see what's "
    "pending). The task is to act on the present situation — `trace_search` "
    "is for INFORMING that action with prior-session memory, not a destination "
    "in itself. Don't summarize the trace or ask the user what to do; find the "
    "pending work in the environment and complete it.\n\n"
    "Complete the task and stop when its success condition is met."
)

TRACE_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "trace_search",
        "description": (
            "Search the prior-session trace (chunked per UTC day). "
            "Returns the top-k days whose events most semantically match the query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": TOP_K_DEFAULT},
            },
            "required": ["query"],
        },
    },
}

async def _exec_with_retries(
    environment: BaseEnvironment,
    command: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run an internal-only shell command (e.g. ``rm -f trace.jsonl``).

    Kept separate from the model-facing tool surface: the LLM never sees
    a ``shell_exec`` tool, so this helper is only used by the agent
    itself for housekeeping (deleting the raw trace post-ingest so the
    only path to prior-session facts is ``trace_search``).
    """
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


@dataclass
class Chunk:
    id: str
    text: str


def _format_event(event: dict[str, Any]) -> str:
    ts = event.get("timestamp", "")
    data = event.get("message_data") or {}
    etype = data.get("type") or "message"
    if etype == "message":
        role = data.get("role") or "user"
        return _truncate_event(f"[{ts}] {role}: {_render_content(data.get('content'))}")
    if etype == "reasoning":
        return _truncate_event(f"[{ts}] reasoning: {_render_content(data.get('summary'))}")
    if etype == "function_call":
        return _truncate_event(
            f"[{ts}] tool_call {data.get('name')!r} "
            f"args={data.get('arguments') or '{}'}"
        )
    if etype == "function_call_output":
        return _truncate_event(
            f"[{ts}] tool_output call_id={data.get('call_id')} "
            f"{data.get('output') or ''}"
        )
    return _truncate_event(f"[{ts}] {json.dumps(data)}")


def _render_content(content: Any) -> str:
    """Render trace content while stripping large opaque fields like signatures."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "thinking":
                parts.append(f"thinking: {item.get('thinking', '')}")
            elif item_type == "summary_text":
                parts.append(str(item.get("text", "")))
            elif item_type in ("tool_use", "function_call"):
                tool_name = item.get("name") or item.get("function", {}).get("name")
                tool_input = item.get("input") or item.get("arguments") or {}
                parts.append(f"tool_use {tool_name}: {json.dumps(tool_input, ensure_ascii=False)}")
            elif item_type == "text":
                parts.append(str(item.get("text", "")))
            else:
                cleaned = {
                    key: value
                    for key, value in item.items()
                    if key not in {"signature", "id"}
                }
                parts.append(json.dumps(cleaned, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        cleaned = {
            key: value
            for key, value in content.items()
            if key not in {"signature", "id"}
        }
        return json.dumps(cleaned, ensure_ascii=False)
    return str(content or "")


def _truncate_event(text: str) -> str:
    if len(text) <= MAX_FORMATTED_EVENT_CHARS:
        return text
    return text[:MAX_FORMATTED_EVENT_CHARS] + "\n[... event truncated for retrieval indexing ...]"


def _chunk_trace_by_day(raw_lines: list[str]) -> list[Chunk]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = event.get("timestamp", "1970-01-01T00:00:00Z")
        try:
            day = (
                datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                .astimezone(UTC)
                .date()
                .isoformat()
            )
        except ValueError:
            day = "undated"
        groups[day].append(_format_event(event))

    chunks: list[Chunk] = []
    for day, lines in sorted(groups.items()):
        part = 1
        current: list[str] = []
        current_len = len(f"# {day}#{part:03d}\n\n")
        for line in lines:
            line_len = len(line) + 1
            if current and current_len + line_len > MAX_CHUNK_CHARS:
                chunk_id = f"{day}#{part:03d}"
                chunks.append(Chunk(id=chunk_id, text=f"# {chunk_id}\n\n" + "\n".join(current)))
                part += 1
                current = []
                current_len = len(f"# {day}#{part:03d}\n\n")
            current.append(line)
            current_len += line_len
        if current:
            chunk_id = f"{day}#{part:03d}"
            chunks.append(Chunk(id=chunk_id, text=f"# {chunk_id}\n\n" + "\n".join(current)))
    return chunks


def _summarize_chunk_ids(chunks: list[Chunk]) -> str:
    if not chunks:
        return "none"
    ids = [c.id for c in chunks]
    if len(ids) <= 12:
        return str(ids)
    return str(ids[:6] + ["..."] + ids[-5:])


class TraceRagAgent(BaseAgent):
    """RAG baseline: per-day chunks over OpenAI embeddings + tool-callable search."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-rag"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceRagAgent requires OPENROUTER_API_KEY "
                "(used for both chat completions and embeddings via OpenRouter)."
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
        call_log: list[dict] = []
        trial_label = f"horizon-trace-rag-{uuid.uuid4().hex[:8]}"

        chunks: list[Chunk] = []
        tool_registry: HorizonToolRegistry | None = None
        steps: list[Step] = []
        total_prompt = total_completion = 0
        chat_cost_usd = 0.0
        ingest_embedding_cost_usd = 0.0
        query_embedding_cost_usd = 0.0
        total_embedding_tokens = 0
        embeddings_cached = False
        t_ingest_done = t_start
        t_end = t_start

        async with trial_subkey(
            management_key=management_key,
            label=trial_label,
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=tk.key,
            )

            # ---- Load the task-specific tool registry from the sandbox ----
            # These are the constrained, eval-author-blessed tools the LLM is
            # allowed to call (no generic shell). We load them up-front so a
            # missing/malformed registry fails the run loudly before we burn
            # any embedding tokens.
            async with timed_call(call_log, "download", "load tools.json"):
                tool_registry = await load_environment_tools(environment)

            # ---- Ingest: read trace, chunk by day, embed each chunk ----
            # Use download_file (not exec/cat) so we don't truncate at the agent's
            # stdout cap — production traces can be tens of MB.
            async with timed_call(call_log, "download", "download trace.jsonl"):
                trace_text = await read_trace_file(environment, TRACE_PATH)
            lines = trace_text.splitlines()
            chunks = _chunk_trace_by_day(lines)

            chunk_ids = [c.id for c in chunks]
            cache_key = _embed_cache_key(chunks, EMBEDDING_MODEL) if chunks else ""
            cached_matrix = (
                _load_embed_cache(cache_key, chunk_ids) if chunks else None
            )

            if cached_matrix is not None:
                # Cache hit: the trace was already embedded by a previous run
                # (e.g. a different chat model over the same task). Skip the
                # embedding round-trips entirely — no tokens, no cost.
                matrix = cached_matrix
                embeddings_cached = True
            elif chunks:
                # Embed batches concurrently via a bounded queue (up to
                # EMBED_INGEST_CONCURRENCY in-flight requests) instead of
                # serially — large traces are thousands of chunks, so serial
                # round-trips dominate wall-clock. Each batch is retried on
                # transient failures; order is preserved so embeddings stay
                # aligned with their chunks.
                batch_starts = list(range(0, len(chunks), EMBEDDING_BATCH_SIZE))
                sem = asyncio.Semaphore(EMBED_INGEST_CONCURRENCY)

                async def _embed_one(start: int):
                    batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
                    texts = [c.text for c in batch]
                    last_exc: Exception | None = None
                    async with sem:
                        for attempt in range(5):
                            try:
                                async with timed_call(
                                    call_log,
                                    "embedding",
                                    f"ingest batch @{start} ({len(batch)} chunks)",
                                ):
                                    return await _create_embeddings_resilient(client, texts)
                            except Exception as exc:  # transient: 429/5xx/socket
                                last_exc = exc
                                await asyncio.sleep(min(2 ** attempt, 16))
                    raise last_exc if last_exc else RuntimeError("embed failed")

                batch_results = await asyncio.gather(
                    *(_embed_one(s) for s in batch_starts)
                )
                embeddings: list[list[float]] = []
                for batch_embeddings, batch_tokens, batch_cost in batch_results:
                    total_embedding_tokens += batch_tokens
                    ingest_embedding_cost_usd += batch_cost
                    embeddings.extend(batch_embeddings)
                if len(embeddings) != len(chunks):
                    raise RuntimeError(
                        "embedding count mismatch: embedded "
                        f"{len(embeddings)} of {len(chunks)} chunks — refusing "
                        "to run with a partially-embedded trace"
                    )
                matrix = np.array(embeddings, dtype=np.float32)
                matrix /= np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
                # Persist the normalized matrix so the next chat model over
                # this same trace is a cache hit.
                _store_embed_cache(cache_key, matrix, chunk_ids)
            else:
                matrix = np.zeros((0, 1), dtype=np.float32)

            # Remove the trace file so the only way to recall prior history is
            # through the embedding-ranked retrieval tool. Best-effort: a Modal
            # stdio blip here is harmless because the LLM is told to use
            # `trace_search`.
            async with timed_call(call_log, "exec", "rm trace.jsonl"):
                await _exec_with_retries(
                    environment, f"rm -f {TRACE_PATH}", timeout_sec=10
                )

            embed_note = (
                f"Loaded cached embeddings ({EMBEDDING_MODEL}, 0 new tokens)"
                if embeddings_cached
                else f"Embedded with {EMBEDDING_MODEL} ({total_embedding_tokens} tokens)"
            )
            ingest_note = (
                f"Ingested {len(chunks)} chunks from {TRACE_PATH}: "
                f"{_summarize_chunk_ids(chunks)}. "
                f"{embed_note}. "
                f"Trace file deleted to force retrieval through `trace_search`."
            )
            self.logger.info(ingest_note)

            # ---- Tool-calling loop ----
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                tool_names=", ".join(f"`{n}`" for n in tool_registry.names) or "(none)"
            )
            user_message = (
                f"Task:\n\n{instruction}\n\n"
                "You have no direct view of the prior-session trace. Call "
                "`trace_search` to retrieve relevant day-chunks before acting."
            )
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            steps.extend(
                [
                    Step(
                        step_id=1,
                        timestamp=_now_iso(),
                        source="system",
                        message=ingest_note,
                    ),
                    Step(
                        step_id=2,
                        timestamp=_now_iso(),
                        source="user",
                        message=user_message,
                    ),
                ]
            )

            t_ingest_done = time.monotonic()

            tools_schema = [TRACE_SEARCH_TOOL, *tool_registry.openrouter_tools]

            for turn_idx in range(MAX_STEPS):
                async with timed_call(call_log, "chat", f"chat turn {turn_idx + 1}"):
                    resp = await client.chat.completions.create(
                        model=chat_model,
                        messages=messages,
                        tools=tools_schema,
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
                    if name == "trace_search":
                        async with timed_call(
                            call_log,
                            "embedding",
                            f"trace_search query (turn {turn_idx + 1})",
                        ):
                            payload, extra_tokens, extra_cost = await self._run_trace_search(
                                client, chunks, matrix, args
                            )
                        total_embedding_tokens += extra_tokens
                        query_embedding_cost_usd += extra_cost
                    elif name in tool_registry:
                        async with timed_call(
                            call_log,
                            "exec",
                            f"{name} turn {turn_idx + 1}",
                        ):
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
                                f"unknown tool: {name}. Available: trace_search, "
                                f"{', '.join(tool_registry.names) or '(none)'}."
                            ),
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

        tool_names = tool_registry.names if tool_registry is not None else []
        direct_total = chat_cost_usd + ingest_embedding_cost_usd + query_embedding_cost_usd
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
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_steps=len(steps),
            ),
            extra={
                "trace_chunks": [c.id for c in chunks],
                "tools": tool_names,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_tokens": total_embedding_tokens,
                "embeddings_cached": embeddings_cached,
                "timing_seconds": {
                    "ingest": round(t_ingest_done - t_start, 3),
                    "chat": round(t_end - t_ingest_done, 3),
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": tk.cost_usd_dict(
                    direct_total=direct_total,
                    breakdown={
                        "ingest_embeddings": round(ingest_embedding_cost_usd, 6),
                        "chat_completions": round(chat_cost_usd, 6),
                        "query_embeddings": round(query_embedding_cost_usd, 6),
                    },
                ),
                "call_summary": summarize_call_log(call_log),
                "call_log": call_log,
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_prompt
        context.n_output_tokens = total_completion

    async def _run_trace_search(
        self,
        client: Any,
        chunks: list[Chunk],
        matrix: np.ndarray,
        args: dict[str, Any],
    ) -> tuple[dict[str, Any], int, float]:
        query = str(args.get("query") or "").strip()
        k = int(args.get("k") or TOP_K_DEFAULT)
        if not query or not chunks:
            return {"hits": [], "note": "empty query or empty trace"}, 0, 0.0

        embeddings, embed_tokens, embed_cost = await _create_embeddings_resilient(
            client, [query]
        )
        q = np.array(embeddings[0], dtype=np.float32)
        q /= np.linalg.norm(q).clip(min=1e-8)

        sims = matrix @ q
        top_idx = np.argsort(-sims)[: min(k, len(chunks))]
        hits = [
            {
                "chunk_id": chunks[i].id,
                "similarity": float(sims[i]),
                "text": chunks[i].text,
            }
            for i in top_idx
        ]
        return {"hits": hits}, embed_tokens, embed_cost


def _embed_cache_dir() -> Path:
    """Directory holding cached ingest-embedding matrices.

    Overridable via ``TRACE_RAG_EMBED_CACHE_DIR`` so multiple machines / runs
    can point at a shared location; defaults to the repo-local ``.cache/``
    convention (gitignored).
    """
    override = os.environ.get("TRACE_RAG_EMBED_CACHE_DIR")
    return Path(override) if override else DEFAULT_EMBED_CACHE_DIR


def _embed_cache_key(chunks: list[Chunk], model: str) -> str:
    """Stable hash over (cache version, embedding model, every chunk id+text).

    Hashing the rendered chunk texts means the key implicitly captures the
    chunking params (``MAX_CHUNK_CHARS``), event formatting/truncation, and
    the trace content itself — anything that would change an embedding input
    changes the key.
    """
    hasher = hashlib.sha256()
    hasher.update(EMBED_CACHE_VERSION.encode())
    hasher.update(b"\x00")
    hasher.update(model.encode())
    for chunk in chunks:
        hasher.update(b"\x00")
        hasher.update(chunk.id.encode())
        hasher.update(b"\x01")
        hasher.update(chunk.text.encode())
    return hasher.hexdigest()


def _load_embed_cache(key: str, expected_ids: list[str]) -> np.ndarray | None:
    """Return the cached L2-normalized matrix for ``key``, or ``None`` on miss.

    Any failure (missing file, partial/corrupt npz, id mismatch) is treated as
    a miss — caching must never be able to fail or poison a run.
    """
    path = _embed_cache_dir() / f"{key}.npz"
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            matrix = data["matrix"]
            cached_ids = [str(i) for i in data["chunk_ids"]]
    except Exception:
        return None
    if cached_ids != expected_ids or matrix.shape[0] != len(expected_ids):
        return None
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _store_embed_cache(key: str, matrix: np.ndarray, chunk_ids: list[str]) -> None:
    """Atomically persist ``matrix`` under ``key`` (best-effort).

    Writes to a unique temp file then ``os.replace`` so concurrent trials
    embedding the same trace can't observe a half-written file. Swallows all
    errors — a failed cache write just means the next run re-embeds.
    """
    cache_dir = _embed_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_dir / f".{key}.{os.getpid()}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(
            tmp,
            matrix=np.ascontiguousarray(matrix, dtype=np.float32),
            chunk_ids=np.array(chunk_ids),
        )
        os.replace(tmp, cache_dir / f"{key}.npz")
    except Exception:
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:
            pass


async def _create_embeddings_resilient(
    client: Any,
    inputs: list[str],
    *,
    attempt: int = 0,
) -> tuple[list[list[float]], int, float]:
    """Create embeddings, splitting batches if OpenRouter drops a response.

    Returns ``(embeddings, total_tokens, total_usd_cost)``. Cost is 0.0 when
    the provider didn't surface a per-call USD figure (the request asks for
    one via ``extra_body={"usage": {"include": True}}``).
    """
    if not inputs:
        return [], 0, 0.0
    try:
        resp = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=inputs,
            extra_body={"usage": {"include": True}},
        )
        data = list(resp.data or [])
        if len(data) != len(inputs):
            raise ValueError(
                f"embedding response length mismatch: got {len(data)}, expected {len(inputs)}"
            )
        tokens = (resp.usage.total_tokens if resp.usage else 0) or 0
        cost = usage_cost(resp)
        return [d.embedding for d in data], tokens, cost
    except Exception:
        if len(inputs) > 1:
            mid = len(inputs) // 2
            left, left_tokens, left_cost = await _create_embeddings_resilient(
                client,
                inputs[:mid],
                attempt=attempt,
            )
            right, right_tokens, right_cost = await _create_embeddings_resilient(
                client,
                inputs[mid:],
                attempt=attempt,
            )
            return left + right, left_tokens + right_tokens, left_cost + right_cost
        if attempt < 2:
            await asyncio.sleep(1 + attempt)
            return await _create_embeddings_resilient(
                client, inputs, attempt=attempt + 1
            )
        raise
