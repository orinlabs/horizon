"""TraceGraphitiAgent — temporal knowledge-graph memory baseline.

Graphiti (getzep/graphiti) builds a *bi-temporal* knowledge graph from a
stream of episodes: it LLM-extracts entities and relationships, embeds
them, and — crucially — when a newly-ingested fact contradicts an older
one it **invalidates the old edge** (marks its ``valid_to``) instead of
silently keeping both. At query time it does a hybrid (semantic + BM25 +
graph-distance) search and can rerank by recency.

That temporal-invalidation behavior is exactly what the recency / override
cases in Horizon-1 stress — ``stale-schedule-vs-actual-pattern``,
``mon-thu-5pm-schedule-correction``, ``test-rescheduled-to-monday``,
``grandfathered-pricing-with-cascading-patches`` — where a flat vector
store (``trace_mem0``) or an append-only memory file (``hermes``) happily
returns *both* the old and the new fact and lets the model pick wrong.

Shape mirrors ``trace_mem0`` so the two are directly comparable:

  - Ingests the FULL trace, grouped by **UTC day**. Each day becomes one
    (or, for very long days, a few) Graphiti episode(s), ingested in
    **chronological order** with ``reference_time`` set to that day so
    the graph's temporal ordering matches the real session timeline.
    (mem0 groups by wake/sleep cycle; Graphiti is ~order-of-magnitude
    more expensive per episode because each add runs extraction +
    edge-resolution LLM calls, so day-grouping keeps the episode count
    — and cost — bounded for multi-month traces.)
  - Exposes ``memory_search`` (``graphiti.search`` → temporal facts) plus
    ``shell_exec`` to act on the environment. No raw-trace access: the
    trace file is deleted from the sandbox after ingest.

Backend: Graphiti's embedded **Kuzu** driver (in-process, no server) on a
per-trial tempdir — the graph analog of mem0's ephemeral Chroma. LLM,
embedder, and reranker are all routed through OpenRouter via the per-trial
sub-key, so the sub-key's ledger captures the *full* per-trial cost
(extraction + embeddings + reranking + chat).

Run it with::

    source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/304-13-stale-schedule-vs-actual-pattern-v0 \\
        --agent-import-path trace_graphiti.agent:TraceGraphitiAgent \\
        -m openai/gpt-4o-mini

Requires ``graphiti-core[kuzu]`` in the host env (``uv add
'graphiti-core[kuzu]'``).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from collections import defaultdict
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
# Graphiti's internal extraction/dedup/reranking model. Fixed + cheap and
# independent of the benchmarked chat model so the built graph is identical
# across a model sweep (same principle as RAG's fixed embedding model).
DEFAULT_GRAPH_MODEL = "openai/gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536
ATIF_VERSION = "ATIF-v1.4"
# Graphiti is given a single default group (group_id=None) per trial. Passing
# an explicit group_id makes Graphiti try to clone the driver onto a new
# "database" named after the group — unsupported on the single-file embedded
# Kuzu backend (it dereferences a `_database` attr KuzuDriver doesn't set, and
# would split the graph across databases anyway). The Kuzu DB is already
# per-trial and ephemeral, so the default group is exactly what we want.
TOP_K_DEFAULT = 5
MAX_FORMATTED_EVENT_CHARS = 800
# Max characters of rendered transcript per episode body. Graphiti extracts
# entities/edges from each episode in one LLM pass, so over-long bodies blow
# the extraction context and degrade quality; we split a day across multiple
# episodes (sharing that day's reference_time) when it exceeds this.
MAX_EPISODE_CHARS = 6_000
# Hard ceiling on episode count so a pathologically long trace can't run the
# (sequential, temporal) ingest past the agent timeout. When exceeded we keep
# the MOST RECENT days (recency is what the override cases care about).
# Overridable via GRAPHITI_MAX_EPISODES (e.g. to bound a quick test run).
MAX_EPISODES = int(os.environ.get("GRAPHITI_MAX_EPISODES", "400"))
MAX_EXEC_OUTPUT_CHARS = 12_000

SYSTEM_PROMPT = (
    "You are an autonomous agent.\n\n"
    "You have tools:\n"
    "  - `memory_search`: query a temporal knowledge graph built from the "
    "prior session. It returns FACTS (graph edges) with validity intervals; "
    "facts that were later corrected/superseded are invalidated, so prefer "
    "currently-valid facts. Query in natural language. You have NO direct "
    "access to the raw trace.\n"
    "  - `shell_exec`: run a shell command in the task environment. The prior "
    "trace shows what shell commands are available (each function_call name "
    "is installed as a shell command of the same name in `/usr/local/bin` "
    "with matching `--flag value` arguments). Use this to act on the current "
    "world.\n\n"
    "Workflow: ALWAYS start with `shell_exec` to inspect the current "
    "environment (e.g. `task_list`, `sms_list`, `show_account` — names match "
    "the trace's function_call events). The task is to act on the present "
    "situation — `memory_search` is for INFORMING that action with the most "
    "up-to-date prior-session facts, not a destination in itself. When facts "
    "conflict, trust the most recent / still-valid one.\n\n"
    "Complete the task and stop when its success condition is met."
)

MEMORY_SEARCH_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": (
            "Hybrid temporal search over the prior session's knowledge graph. "
            "Returns currently-relevant facts (edges) with validity dates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language query."},
                "k": {
                    "type": "integer",
                    "description": f"Facts to return (default {TOP_K_DEFAULT}).",
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

# (kuzu_table, fts_index_name, [indexed_columns]) — the 4 indices Graphiti's
# KuzuSearchOperations queries. KuzuDriver.setup_schema() creates the tables
# but NOT these FTS indices (getzep/graphiti#1360), so we build them by hand
# or every search raises "Table X doesn't have an index Y".
_KUZU_FTS_DEFS = [
    ("Episodic", "episode_content", ["content", "source", "source_description"]),
    ("Entity", "node_name_and_summary", ["name", "summary"]),
    ("Community", "community_name", ["name"]),
    ("RelatesToNode_", "edge_name_and_fact", ["name", "fact"]),
]


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


def _event_day(line: str) -> str:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return "undated"
    ts = obj.get("timestamp", "")
    try:
        return (
            datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            .astimezone(UTC)
            .date()
            .isoformat()
        )
    except (ValueError, TypeError):
        return "undated"


def _day_reference_time(day: str) -> datetime:
    """UTC midnight for a YYYY-MM-DD day key; falls back to now() for undated."""
    try:
        return datetime.fromisoformat(day).replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


def _build_episodes(all_lines: list[str]) -> list[tuple[str, str, datetime]]:
    """Group trace lines into (name, body, reference_time) episodes by UTC day.

    Days are emitted oldest-first so chronological ingest preserves Graphiti's
    temporal ordering (later facts can invalidate earlier ones). Long days are
    split into multiple part-episodes that share the day's reference_time. If
    the trace produces more than MAX_EPISODES episodes we keep the most recent
    days (recency dominates the override cases).
    """
    by_day: dict[str, list[str]] = defaultdict(list)
    for line in all_lines:
        if line.strip():
            by_day[_event_day(line)].append(line)

    # Sort real dates ascending; pin "undated" first so it can't claim recency.
    days = sorted(by_day, key=lambda d: (d != "undated", d))

    episodes: list[tuple[str, str, datetime]] = []
    for day in days:
        ref = _day_reference_time(day)
        rendered = [_format_event(line) for line in by_day[day]]
        buf: list[str] = []
        size = 0
        part = 0
        for piece in rendered:
            if buf and size + len(piece) > MAX_EPISODE_CHARS:
                part += 1
                episodes.append(
                    (f"session {day} (part {part})", "\n".join(buf), ref)
                )
                buf, size = [], 0
            buf.append(piece)
            size += len(piece) + 1
        if buf:
            part += 1
            name = f"session {day}" if part == 1 else f"session {day} (part {part})"
            episodes.append((name, "\n".join(buf), ref))

    if len(episodes) > MAX_EPISODES:
        episodes = episodes[-MAX_EPISODES:]
    return episodes


async def _subkey_usage_usd(key: str) -> float:
    """Total USD spent on an OpenRouter sub-key (best-effort ledger read).

    Graphiti makes many internal LLM/embedding calls we don't see per-response,
    so — like the hermes agent — we read the sub-key's full ledger to capture
    the real per-trial cost (extraction + embeddings + reranking + chat).
    """
    if not key:
        return 0.0
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {key}"},
            )
            if resp.status_code != 200:
                return 0.0
            return float((resp.json().get("data") or {}).get("usage") or 0.0)
    except Exception:
        return 0.0


def _build_graphiti(db_path: str, api_key: str) -> Any:
    """Construct a Graphiti pointed at OpenRouter over an embedded Kuzu graph."""
    from graphiti_core import Graphiti
    from graphiti_core.cross_encoder.openai_reranker_client import (
        OpenAIRerankerClient,
    )
    from graphiti_core.driver.kuzu_driver import KuzuDriver
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient

    graph_model = os.environ.get("GRAPHITI_GRAPH_MODEL", DEFAULT_GRAPH_MODEL)
    embedding_model = os.environ.get("GRAPHITI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    base_url = "https://openrouter.ai/api/v1"

    llm_config = LLMConfig(
        api_key=api_key,
        model=graph_model,
        small_model=graph_model,
        base_url=base_url,
    )
    # The standard OpenAIClient (structured-output extraction) materially
    # out-extracts OpenAIGenericClient over OpenRouter->OpenAI models — in
    # particular it captures temporal "scheduled for X" facts the generic
    # client drops, which are exactly what this agent needs. (Generic is only
    # preferable for providers without json_schema support, e.g. local Ollama.)
    llm_client = OpenAIClient(config=llm_config)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=api_key,
            embedding_model=embedding_model,
            embedding_dim=EMBEDDING_DIM,
            base_url=base_url,
        )
    )
    # Reuse the LLM client for reranking (boolean-classification reranker) so
    # we don't need a separate cross-encoder dependency or a model that exposes
    # raw logprobs.
    cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)

    driver = KuzuDriver(db=db_path)
    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


async def _ensure_kuzu_fts_indices(graphiti: Any, logger: Any) -> None:
    """Create the FTS indices KuzuDriver.setup_schema() omits (issue #1360).

    Best-effort: loads the FTS extension (newer Kuzu ships it; LOAD is a no-op
    if already present) and creates each index, ignoring "already exists"
    errors. If this fails, search will error at query time but ingest and the
    shell-acting path still work.
    """
    driver = graphiti.driver
    for stmt in ("INSTALL FTS;", "LOAD FTS;"):
        try:
            await driver.execute_query(stmt)
        except Exception:
            pass
    for table, index, cols in _KUZU_FTS_DEFS:
        col_list = ", ".join(f"'{c}'" for c in cols)
        try:
            await driver.execute_query(
                f"CALL CREATE_FTS_INDEX('{table}', '{index}', [{col_list}])"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("kuzu FTS index %s.%s create skipped: %s", table, index, exc)


async def _rebuild_kuzu_fts_indices(graphiti: Any, logger: Any) -> None:
    """Drop + recreate FTS indices after ingest.

    Kuzu FTS indices are static snapshots over the rows present at creation
    time, so episodes added after the initial CREATE aren't searchable until
    the index is rebuilt. We create indices before ingest (Graphiti's internal
    entity-resolution searches need them) and rebuild here so the agent-facing
    search sees the whole graph.
    """
    driver = graphiti.driver
    for table, index, cols in _KUZU_FTS_DEFS:
        col_list = ", ".join(f"'{c}'" for c in cols)
        try:
            await driver.execute_query(f"CALL DROP_FTS_INDEX('{table}', '{index}')")
        except Exception:
            pass
        try:
            await driver.execute_query(
                f"CALL CREATE_FTS_INDEX('{table}', '{index}', [{col_list}])"
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("kuzu FTS index %s.%s rebuild skipped: %s", table, index, exc)


class TraceGraphitiAgent(BaseAgent):
    """Temporal knowledge-graph memory baseline (Graphiti + embedded Kuzu)."""

    SUPPORTS_ATIF = True

    @staticmethod
    def name() -> str:
        return "trace-graphiti"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError(
                "TraceGraphitiAgent requires OPENROUTER_API_KEY in the host env."
            )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from graphiti_core.nodes import EpisodeType

        from openai import AsyncOpenAI

        management_key = os.environ["OPENROUTER_MANAGEMENT_KEY"]
        chat_model = self.model_name or DEFAULT_CHAT_MODEL

        t_start = time.monotonic()
        trial_label = f"horizon-trace-graphiti-{uuid.uuid4().hex[:8]}"

        total_pt = total_ct = 0
        chat_cost_usd = 0.0
        n_episodes = 0
        n_episodes_ok = 0
        ingest_seconds = 0.0
        all_lines: list[str] = []
        searches_done: list[dict[str, Any]] = []
        steps: list[Step] = []
        t_ingest_done = t_start
        t_end = t_start
        trial_total_cost_usd = 0.0

        async with trial_subkey(
            management_key=management_key,
            label=trial_label,
        ) as tk:
            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1", api_key=tk.key
            )

            trace_text = await read_trace_file(environment, TRACE_PATH)
            all_lines = trace_text.splitlines() if trace_text else []
            episodes = _build_episodes(all_lines)
            n_episodes = len(episodes)

            # Force the model through memory_search by removing the raw trace.
            await _exec_with_retries(environment, f"rm -f {TRACE_PATH}", timeout_sec=10)

            db_dir = tempfile.mkdtemp(prefix="trace_graphiti_kuzu_")
            db_path = os.path.join(db_dir, "graphiti.kuzu")
            graphiti = None
            try:
                graphiti = _build_graphiti(db_path, tk.key)
                # Build base schema (Neo4j-only no-op for Kuzu) then the FTS
                # indices Kuzu needs before any search (incl. ingest-internal).
                try:
                    await graphiti.build_indices_and_constraints()
                except Exception as exc:  # noqa: BLE001
                    self.logger.debug("build_indices_and_constraints skipped: %s", exc)
                await _ensure_kuzu_fts_indices(graphiti, self.logger)

                ingest_started = time.monotonic()
                # Sequential, chronological ingest. add_episode mutates the graph
                # and resolves edges against existing nodes; out-of-order or
                # concurrent adds would corrupt the temporal ordering that is
                # this agent's whole point. Slow but faithful.
                for name, body, ref_time in episodes:
                    try:
                        await graphiti.add_episode(
                            name=name,
                            episode_body=body,
                            source=EpisodeType.text,
                            source_description="prior agent session transcript",
                            reference_time=ref_time,
                            group_id=None,
                        )
                        n_episodes_ok += 1
                    except Exception as exc:
                        self.logger.warning(
                            "graphiti add_episode failed (%s): %s: %s",
                            name,
                            type(exc).__name__,
                            exc,
                        )
                ingest_seconds = time.monotonic() - ingest_started
                await _rebuild_kuzu_fts_indices(graphiti, self.logger)
                self.logger.info(
                    "graphiti ingest: %d lines -> %d/%d episodes in %.1fs",
                    len(all_lines),
                    n_episodes_ok,
                    n_episodes,
                    ingest_seconds,
                )

                t_ingest_done = time.monotonic()

                all_tools = [MEMORY_SEARCH_TOOL, SHELL_EXEC_TOOL]
                user_message = (
                    f"You have access to a temporal knowledge graph built from a "
                    f"{len(all_lines)}-event prior session ({n_episodes_ok} episodes "
                    f"ingested chronologically). Query it via `memory_search`; the "
                    f"raw trace is no longer accessible.\n\nCurrent task:\n\n{instruction}"
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
                                hits = await graphiti.search(
                                    query=query,
                                    group_ids=None,
                                    num_results=k,
                                )
                            except Exception as exc:
                                payload = {
                                    "exit_code": 1,
                                    "stdout": "",
                                    "stderr": f"graphiti search failed: {exc!r}",
                                }
                            else:
                                facts = []
                                for edge in hits or []:
                                    facts.append(
                                        {
                                            "fact": getattr(edge, "fact", str(edge)),
                                            "valid_at": _edge_dt(edge, "valid_at"),
                                            "invalid_at": _edge_dt(edge, "invalid_at"),
                                        }
                                    )
                                payload = {
                                    "exit_code": 0,
                                    "stdout": json.dumps(facts),
                                    "stderr": "",
                                }
                                searches_done.append(
                                    {"query": query, "k": k, "n_hits": len(facts)}
                                )
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
                if graphiti is not None:
                    try:
                        await graphiti.close()
                    except Exception:
                        pass
                shutil.rmtree(db_dir, ignore_errors=True)

            # Read the sub-key's full ledger before it's deleted: this captures
            # Graphiti's internal extraction/embedding/rerank spend that the
            # per-response usage_cost above can't see.
            trial_total_cost_usd = await _subkey_usage_usd(tk.key)

        # Fall back to the chat-only accounting if the ledger read came back 0.
        direct_total = trial_total_cost_usd or chat_cost_usd
        ingest_cost = max(0.0, round(direct_total - chat_cost_usd, 6))

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
                "episodes_total": n_episodes,
                "episodes_ingested": n_episodes_ok,
                "ingest_seconds": round(ingest_seconds, 2),
                "searches": searches_done,
                "timing_seconds": {
                    "ingest": round(t_ingest_done - t_start, 3),
                    "chat": round(t_end - t_ingest_done, 3),
                    "total": round(t_end - t_start, 3),
                },
                "cost_usd": tk.cost_usd_dict(
                    direct_total=round(direct_total, 6),
                    breakdown={
                        "chat": round(chat_cost_usd, 6),
                        "graph_build": ingest_cost,
                    },
                ),
            },
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )

        context.n_input_tokens = total_pt
        context.n_output_tokens = total_ct


def _edge_dt(edge: Any, attr: str) -> str | None:
    """ISO-format an optional datetime attribute on a Graphiti edge."""
    value = getattr(edge, attr, None)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
