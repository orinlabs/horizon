"""HermesAgent — Harbor wrapper around NousResearch/hermes-agent.

Installs hermes via its upstream ``install.sh`` (the same recipe their
published Docker image uses), then seeds past sessions into Hermes's native
SessionDB from ``/workdir/trace.jsonl`` so the agent's built-in
``session_search`` tool can recall the prior history, then runs hermes in
one-shot programmatic mode. Emits an ATIF-compliant ``trajectory.json`` by
reading the messages hermes persists to ``$HERMES_HOME/state.db`` during the
run.

Run it with::

    source .env && export OPENROUTER_API_KEY
    PYTHONPATH=agents harbor run \\
        -p evals/therapy-goals-followthrough \\
        --agent-import-path hermes.agent:HermesAgent \\
        -m anthropic/claude-sonnet-4.6

First install is slow (~5 min) because hermes pulls uv, Python 3.11, and
Node.js 22 into the task container.  Harbor caches the Docker layer for the
task image itself, so only the first-ever run against a brand-new task image
pays the full cost; per-trial startup re-runs ``install.sh`` which no-ops when
already installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from agent_utils import TrialKeyState, trial_subkey
from harbor.agents.installed.base import (
    BaseInstalledAgent,
    EnvVar,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Step,
    Trajectory,
)


INSTALL_SH = (
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh"
)
HERMES_HOME = "/root/.hermes"
HERMES_VENV_PY = "/usr/local/lib/hermes-agent/venv/bin/python"
TRACE_PATH = "/workdir/trace.jsonl"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
# Fixed, cheap model used for the per-day memory-flush replay (the "build"
# step). Kept independent of the task model so memory is identical across the
# benchmarked acting models — same principle as RAG's fixed embedding model.
MEMORY_MODEL = "openai/gpt-5-mini"
ATIF_VERSION = "ATIF-v1.4"
# Max transcript chars per memory-build LLM call.
MEMORY_BUILD_CHUNK_CHARS = 60_000
# Host-side cache of built ~/.hermes memory, keyed by trace content hash.
# Overridable so a shared location can be reused across machines.
DEFAULT_HERMES_MEM_CACHE_DIR = Path(".cache/hermes_memory")
SANDBOX_MEM_TAR = "/tmp/hermes_mem.tgz"

# Per-trace-hash locks so concurrent trials sharing a trace build memory once;
# the rest wait and restore from cache. In-process (one harbor run = one
# asyncio loop), which is exactly the fill-pass case.
_HERMES_MEM_LOCKS: dict[str, asyncio.Lock] = {}
_HERMES_MEM_LOCKS_GUARD = asyncio.Lock()


async def _hermes_mem_lock(key: str) -> asyncio.Lock:
    async with _HERMES_MEM_LOCKS_GUARD:
        lock = _HERMES_MEM_LOCKS.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _HERMES_MEM_LOCKS[key] = lock
        return lock


def _hermes_mem_cache_dir() -> Path:
    override = os.environ.get("HERMES_MEMORY_CACHE_DIR")
    return Path(override) if override else DEFAULT_HERMES_MEM_CACHE_DIR


def _hermes_cost_sidecar(trace_hash: str) -> Path:
    # Per-trace memory-build cost, stored next to the cached memory tarball so
    # it's attributed to EVERY trial of that case (cache hit or miss) — a real
    # cost any standalone Hermes run would pay before acting.
    return _hermes_mem_cache_dir() / f"{trace_hash}.cost.json"


async def _subkey_usage_usd(key: str) -> float:
    """Total USD spent on an OpenRouter sub-key (best-effort).

    `hermes chat` runs as a subprocess so we can't sum per-response usage like
    the RAG agent does; instead we read the sub-key's ledger via GET /key.
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


def _parse_membuild_result(stdout: str) -> dict:
    """Pull the MEMBUILD_RESULT json line emitted by MEMORY_BUILD_PY."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("MEMBUILD_RESULT "):
            try:
                return json.loads(line[len("MEMBUILD_RESULT "):])
            except json.JSONDecodeError:
                return {}
    return {}


# Build Hermes's memory from the trace WITHOUT running the full `hermes chat`
# agent (which is ~10x slower: it stuffs a 53KB skills snapshot + manual + 28
# plugins into every prompt and runs a multi-turn loop). Instead we drive only
# the memory step: per day-chunk, one lean tool-calling LLM request using
# Hermes's REAL `memory` tool, applied against Hermes's REAL MemoryStore — so
# MEMORY.md / USER.md are written by Hermes's own writer, format, and capacity
# rules. We also seed SessionDB directly (no LLM) so `session_search` works.
MEMORY_BUILD_PY = r'''
import json, os, sys, time, uuid
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "/usr/local/lib/hermes-agent")
from tools.memory_tool import MemoryStore, memory_tool, MEMORY_SCHEMA
try:
    from hermes_state import SessionDB
except Exception:
    SessionDB = None
from openai import OpenAI

trace_path, model, max_chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])

# Prefer a direct OpenAI key when present: OpenAI's gpt-5-mini limits (30k RPM /
# 180M TPM) dwarf OpenRouter's, so high concurrency doesn't get throttled. Fall
# back to OpenRouter otherwise. Model is fixed (MEMORY_MODEL) either way so the
# cached memory is identical across the benchmarked models.
# Tight per-call timeout + no SDK-internal retries: a hung connection otherwise
# blocks ~10 min (SDK default) x internal retries, and our backoff loop below
# multiplies that into hours stuck on one chunk. Fail fast, let our loop retry.
_OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if _OPENAI_KEY:
    USE_OPENAI = True
    client = OpenAI(api_key=_OPENAI_KEY, timeout=90.0, max_retries=0)
    if model.startswith("openai/"):
        model = model.split("/", 1)[1]  # OpenAI wants "gpt-5-mini", not "openai/..."
else:
    USE_OPENAI = False
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    timeout=90.0, max_retries=0)

# Per-1M-token (input, output) USD prices for token-based costing on the direct
# OpenAI path (OpenAI doesn't attach usage.cost like OpenRouter does). Output
# price covers reasoning tokens too (billed as completion).
_OPENAI_PRICE = {"gpt-5-mini": (0.25, 2.0)}


def usage_cost(resp):
    # Mirror agent_utils.usage_cost (not importable in-sandbox): read the USD
    # cost OpenRouter attaches when extra_body usage.include=True. BYOK sub-keys
    # report cost=0, so fall back to cost_details.upstream_inference_cost.
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    dumped = usage.model_dump() if hasattr(usage, "model_dump") else {}
    cost = dumped.get("cost")
    if (cost is None or cost == 0) and dumped.get("is_byok") and dumped.get("cost_details"):
        cost = dumped["cost_details"].get("upstream_inference_cost")
    try:
        return float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def role_and_line(event):
    d = event.get("message_data") or {}
    t = d.get("type")
    if t == "message":
        r = d.get("role") or "user"
        return r, f"{r}: {d.get('content') or ''}"
    if t == "reasoning":
        return "assistant", "assistant: [reasoning] " + str(d.get("summary") or "")
    if t == "function_call":
        return "assistant", f"assistant: [tool:{d.get('name') or 'tool'}] {d.get('arguments') or '{}'}"
    if t == "function_call_output":
        return "tool", "tool: " + str(d.get("output") or "")
    return None, None


groups = defaultdict(list)
with open(trace_path) as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = ev.get("timestamp", "1970-01-01T00:00:00Z")
        try:
            day = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc).date().isoformat()
        except ValueError:
            day = "undated"
        role, line = role_and_line(ev)
        if line and line.strip():
            groups[day].append((role, line))
order = sorted(groups)

# Seed SessionDB for session_search (cheap, no LLM calls).
if SessionDB is not None:
    try:
        db = SessionDB()
        for day in order:
            sid = f"horizon-seed-{day}-{uuid.uuid4().hex[:8]}"
            db.create_session(session_id=sid, source="cli", model=model)
            for role, line in groups[day]:
                db.append_message(session_id=sid, role=role, content=line)
        db.close()
    except Exception as e:
        print("sessiondb seed failed:", e)

store = MemoryStore()
store.load_from_disk()


def current_memory():
    parts = [store.format_for_system_prompt(t) for t in ("memory", "user")]
    return "\n\n".join(p for p in parts if p) or "(empty)"


SYSTEM = (
    "You maintain long-term memory across sessions using the `memory` tool. "
    "You are shown transcripts of PAST sessions, oldest first. For each, call "
    "the `memory` tool to save durable facts, user preferences, standing "
    "policies/rules, names, and environment details that will matter later. "
    "Replace/consolidate when near the limit. Do not take any other action and "
    "do not reply with prose -- only emit memory tool calls."
)
tools = [{"type": "function", "function": MEMORY_SCHEMA}]

seen = set()  # dedupe exact-duplicate lines (collapses repeated doc dumps)
calls = 0
total_cost = 0.0
for day in order:
    lines = []
    for _role, line in groups[day]:
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    text = "\n".join(lines).strip()
    for i in range(0, len(text), max_chunk):
        chunk = text[i:i + max_chunk]
        messages = [
            {"role": "system", "content": SYSTEM + "\n\nCURRENT MEMORY:\n" + current_memory()},
            {"role": "user", "content": f"Past session ({day}):\n\n{chunk}"},
        ]
        kwargs = dict(model=model, messages=messages, tools=tools)
        if USE_OPENAI:
            pass  # gpt-5-mini is a reasoning model: omit temperature (only default supported)
        else:
            kwargs["temperature"] = 0
            kwargs["extra_body"] = {"usage": {"include": True}}
        resp = None
        for attempt in range(8):
            try:
                resp = client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                # Retry transient failures (esp. 429s under high concurrency) with
                # exponential backoff so chunks are never silently dropped.
                wait = min(90, 3 * (2 ** attempt))
                print(f"llm call failed (attempt {attempt + 1}/8): {e}; retry in {wait}s", flush=True)
                time.sleep(wait)
        if resp is None:
            print("llm call permanently failed; skipping chunk", flush=True)
            continue
        calls += 1
        if USE_OPENAI:
            u = resp.usage
            pin, pout = _OPENAI_PRICE.get(model, (0.25, 2.0))
            if u:
                total_cost += ((u.prompt_tokens or 0) / 1e6 * pin
                               + (u.completion_tokens or 0) / 1e6 * pout)
        else:
            total_cost += usage_cost(resp)
        for tc in (resp.choices[0].message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                continue
            memory_tool(
                action=args.get("action", ""), target=args.get("target", "memory"),
                content=args.get("content"), old_text=args.get("old_text"), store=store,
            )

store.save_to_disk("memory")
store.save_to_disk("user")
# Final line is machine-parsed by HermesAgent.install() to record the per-case
# memory-build cost (a real cost any standalone Hermes run would pay, even
# though we cache + reuse the built memory across the model sweep).
print("MEMBUILD_RESULT " + json.dumps({
    "calls": calls,
    "memory_build_cost_usd": round(total_cost, 6),
    "memory_model": model,
    "mem_chars": store._char_count("memory"),
    "user_chars": store._char_count("user"),
}))
'''


DUMP_PY = r'''
"""Dump the most recent Hermes CLI session as JSON for ATIF conversion."""
import json, os, sys, sqlite3

home = os.environ.get("HERMES_HOME", "/root/.hermes")
db_path = os.path.join(home, "state.db")
if not os.path.exists(db_path):
    json.dump({"session": None, "messages": []}, sys.stdout)
    sys.exit(0)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

session = conn.execute(
    """
    SELECT id, source, model, started_at
    FROM sessions
    WHERE source = 'cli' AND id NOT LIKE 'horizon-seed-%'
    ORDER BY started_at DESC
    LIMIT 1
    """
).fetchone()
if session is None:
    json.dump({"session": None, "messages": []}, sys.stdout)
    sys.exit(0)

messages = [
    dict(row)
    for row in conn.execute(
        """
        SELECT id, role, content, tool_calls, tool_name, token_count, timestamp
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session["id"],),
    ).fetchall()
]
json.dump({"session": dict(session), "messages": messages}, sys.stdout)
'''


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _to_iso(value: object) -> str:
    """Coerce whatever Hermes's state.db produced (epoch float, int, or str)
    into an ISO-8601 UTC string. ATIF requires a string timestamp."""
    if value is None:
        return _now_iso()
    if isinstance(value, (int, float)):
        return (
            datetime.fromtimestamp(float(value), tz=UTC)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
    return str(value)


class HermesAgent(BaseInstalledAgent):
    """Harbor-native wrapper around `hermes` (NousResearch/hermes-agent)."""

    SUPPORTS_ATIF = True

    ENV_VARS = [
        EnvVar(
            kwarg="openrouter_api_key",
            env="OPENROUTER_API_KEY",
            env_fallback="OPENROUTER_API_KEY",
        ),
    ]

    @staticmethod
    def name() -> str:
        return "hermes"

    def get_version_command(self) -> str | None:
        return "hermes --version 2>&1 | head -1"

    async def install(self, environment: BaseEnvironment) -> None:
        if "OPENROUTER_API_KEY" not in self._resolved_env_vars:
            raise RuntimeError(
                "HermesAgent requires OPENROUTER_API_KEY; set it in your host env "
                "before invoking `harbor run`."
            )
        model = self.model_name or DEFAULT_MODEL

        # Fast path: detect a pre-existing install from a mounted cache volume
        # (see README for the `--mounts-json` invocation that enables this).
        # install.sh is idempotent but still spends 1-2 min verifying; skipping
        # it entirely when the venv is present drops warm-start setup to ~5s.
        probe = await environment.exec(
            f"test -x {HERMES_VENV_PY} && "
            f"test -x /usr/local/lib/hermes-agent/venv/bin/hermes",
            user="root",
        )
        already_installed = probe.return_code == 0

        if not already_installed:
            # System prereqs pulled from NousResearch/hermes-agent's own
            # Dockerfile (debian:13.4). xz-utils is what install.sh silently
            # needs to unpack its vendored Node.js tarball.
            await self.exec_as_root(
                environment,
                (
                    "apt-get update && "
                    "apt-get install -y --no-install-recommends "
                    "curl ca-certificates git xz-utils build-essential "
                    "python3 python3-venv python3-dev libffi-dev procps"
                ),
                timeout_sec=600,
            )

            # When /usr/local/lib/hermes-agent is a mount point to an empty
            # host directory (warm-cache scaffolding — see README), install.sh
            # refuses to write into a non-git directory. Pre-clone the repo so
            # install.sh takes the "existing install, update" code path.
            await self.exec_as_root(
                environment,
                (
                    "INSTALL_DIR=/usr/local/lib/hermes-agent; "
                    'if [ -d "$INSTALL_DIR" ] && [ ! -d "$INSTALL_DIR/.git" ] && '
                    '   [ -z "$(ls -A "$INSTALL_DIR" 2>/dev/null)" ]; then '
                    "  git clone --depth 1 "
                    "    https://github.com/NousResearch/hermes-agent.git "
                    "    /tmp/hermes-clone && "
                    "  (shopt -s dotglob && mv /tmp/hermes-clone/* \"$INSTALL_DIR/\") && "
                    "  rmdir /tmp/hermes-clone; "
                    "fi"
                ),
                timeout_sec=120,
            )

            # Run hermes's own install.sh in non-interactive mode.
            await self.exec_as_root(
                environment,
                f"curl -fsSL {INSTALL_SH} | bash -s -- --skip-setup",
                timeout_sec=1200,
            )
        else:
            # Warm path: the venv is mounted in. Only the /usr/local/bin/hermes
            # symlink needs recreating because /usr/local/bin is not on the
            # cache volume.
            await self.exec_as_root(
                environment,
                "ln -sf /usr/local/lib/hermes-agent/venv/bin/hermes "
                "/usr/local/bin/hermes",
            )

        # Minimal config: point at OpenRouter + keep default memory on.
        # flush_min_turns: 1 makes Hermes run its memory flush after even a
        # single-turn session, so each per-day replay below triggers the real
        # "save important memories" step before the session ends.
        config_yaml = (
            "memory:\n"
            "  memory_enabled: true\n"
            "  user_profile_enabled: true\n"
            "  nudge_interval: 10\n"
        )
        await self.exec_as_root(
            environment,
            f"mkdir -p {HERMES_HOME} && "
            f"cat > {HERMES_HOME}/config.yaml <<'HERMES_CFG_EOF'\n"
            f"{config_yaml}HERMES_CFG_EOF",
        )

        # Build memory by driving ONLY Hermes's memory writer (not a full
        # `hermes chat` agent loop): one lean tool-calling LLM request per
        # day-chunk against Hermes's real `memory` tool + MemoryStore. Also
        # seeds SessionDB for `session_search`. Fixed cheap model so the built
        # memory is identical across the benchmarked task models — which makes
        # it cacheable per trace and reusable across every model.
        # Per-case memory-build cost (USD). Stays 0.0 if there's no trace or no
        # cost was recorded; otherwise set from the cache sidecar (hit) or the
        # build's own usage accounting (miss). populate_context_post_run folds
        # it into the trial's reported cost.
        self._mem_build_cost_usd = 0.0
        self._mem_build_meta: dict = {}

        if "OPENROUTER_API_KEY" not in self._resolved_env_vars:
            return

        # Hash the trace (content-addressed cache key).
        probe = await environment.exec(
            f"test -f {TRACE_PATH} && sha256sum {TRACE_PATH} | cut -d' ' -f1 || echo NONE",
            user="root",
        )
        out = (probe.stdout or "").strip().splitlines()
        trace_hash = out[-1].strip() if out else "NONE"
        if not trace_hash or trace_hash == "NONE":
            return  # no trace → nothing to build

        cache_tar = _hermes_mem_cache_dir() / f"{trace_hash}.tgz"

        # Per-hash lock: first trial for this trace builds; the rest restore.
        lock = await _hermes_mem_lock(trace_hash)
        async with lock:
            if cache_tar.exists():
                # Cache hit: push memory + session DB into the sandbox; skip build.
                await environment.upload_file(cache_tar, SANDBOX_MEM_TAR)
                await self.exec_as_root(
                    environment,
                    f"mkdir -p {HERMES_HOME} && tar xzf {SANDBOX_MEM_TAR} -C {HERMES_HOME}",
                )
                # Attribute the recorded per-case build cost even on a hit.
                sidecar = _hermes_cost_sidecar(trace_hash)
                if sidecar.exists():
                    try:
                        self._mem_build_meta = json.loads(sidecar.read_text())
                        self._mem_build_cost_usd = float(
                            self._mem_build_meta.get("memory_build_cost_usd") or 0.0
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.logger.warning("hermes cost sidecar read failed: %s", exc)
                self.logger.info(
                    "hermes memory: cache HIT %s (build $%.4f)",
                    trace_hash[:12], self._mem_build_cost_usd,
                )
                return

            # Cache miss: FAITHFUL build in the sandbox — drive Hermes's REAL
            # AIAgent + ContextCompressor (50% compaction) + background_review
            # (memory review every 10 wake-turns) over the trace as one
            # continuous context, using the baked Hermes venv. Then pull the
            # result (~/.hermes) to the host cache.
            builder_host = (
                Path(__file__).resolve().parents[2]
                / "scripts" / "build_hermes_memory_faithful.py"
            )
            await environment.upload_file(builder_host, "/tmp/faithful.py")
            build_env = dict(self._resolved_env_vars)
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("faithful Hermes build requires OPENAI_API_KEY on the host")
            build_env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
            build_env["HERMES_HOME"] = HERMES_HOME
            build_log = f"{HERMES_HOME}/faithful_build.log"
            # Launch the builder DETACHED, streaming progress to a log file, so
            # we can poll it for INCREMENTAL progress (the builder flushes a line
            # per review/compaction). A single blocking exec would hide all
            # progress until the very end.
            launch = (
                f"mkdir -p {HERMES_HOME} && cd /tmp && "
                f"HERMES_HOME={HERMES_HOME} setsid nohup {HERMES_VENV_PY} /tmp/faithful.py insandbox "
                f"--trace {TRACE_PATH} --model {shlex.quote(model)} --nudge 10 "
                f"< /dev/null > {build_log} 2>&1 & echo PID:$!"
            )
            launch_res = await self.exec_as_root(environment, launch, env=build_env)
            pid = ""
            for ln in (launch_res.stdout or "").splitlines():
                if ln.startswith("PID:"):
                    pid = ln[4:].strip()
            self.logger.info("hermes faithful build started (pid %s) %s", pid, trace_hash[:12])

            # Poll the build log for incremental progress until it emits
            # MEMBUILD_RESULT or the process exits. ~20s cadence; cap well under
            # the agent setup timeout.
            last_line = ""
            build_stdout = ""
            for _poll in range(1080):  # ~6h ceiling @ 20s (exits early on MEMBUILD_RESULT)
                await asyncio.sleep(20)
                chk = await environment.exec(
                    f"tail -n 1 {build_log} 2>/dev/null; echo '@@MARK@@'; "
                    f"(kill -0 {pid} 2>/dev/null && echo ALIVE || echo EXITED); "
                    f"grep -q MEMBUILD_RESULT {build_log} 2>/dev/null && echo HASRESULT || true",
                    user="root",
                )
                out = chk.stdout or ""
                head, _, tail = out.partition("@@MARK@@")
                prog = head.strip().splitlines()[-1] if head.strip() else ""
                if prog and prog != last_line:
                    self.logger.info("hermes build [%s] %s", trace_hash[:8], prog)
                    last_line = prog
                if "HASRESULT" in tail or "EXITED" in tail:
                    break

            full = await environment.exec(f"cat {build_log} 2>/dev/null", user="root")
            build_stdout = full.stdout or ""
            try:
                (self.logs_dir / "faithful-build.log").write_text(build_stdout)
            except Exception:
                pass
            self._mem_build_meta = _parse_membuild_result(build_stdout)
            self._mem_build_cost_usd = float(
                self._mem_build_meta.get("memory_build_cost_usd") or 0.0
            )
            self.logger.info(
                "hermes faithful build complete: %s", json.dumps(self._mem_build_meta)
            )
            # Tar memories + session DB (WAL files included for a consistent copy).
            await self.exec_as_root(
                environment,
                f"cd {HERMES_HOME} && tar czf {SANDBOX_MEM_TAR} memories "
                f"state.db state.db-wal state.db-shm 2>/dev/null "
                f"|| tar czf {SANDBOX_MEM_TAR} memories 2>/dev/null || true",
            )
            cache_dir = _hermes_mem_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_tar = cache_dir / f".{trace_hash}.{uuid.uuid4().hex}.tmp"
            try:
                await environment.download_file(SANDBOX_MEM_TAR, tmp_tar)
                os.replace(tmp_tar, cache_tar)  # atomic publish
                # Persist the per-case build cost beside the tarball so every
                # later (cache-hit) trial of this case can attribute it.
                try:
                    sidecar = _hermes_cost_sidecar(trace_hash)
                    tmp_cost = cache_dir / f".{trace_hash}.{uuid.uuid4().hex}.cost.tmp"
                    tmp_cost.write_text(json.dumps(self._mem_build_meta))
                    os.replace(tmp_cost, sidecar)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("hermes cost sidecar save failed: %s", exc)
                self.logger.info(
                    "hermes memory: built + cached %s (build $%.4f, %s calls)",
                    trace_hash[:12], self._mem_build_cost_usd,
                    self._mem_build_meta.get("calls"),
                )
            except Exception as exc:
                self.logger.warning("hermes memory cache save failed: %s", exc)
                try:
                    tmp_tar.unlink(missing_ok=True)
                except Exception:
                    pass

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or DEFAULT_MODEL
        management_key = os.environ["OPENROUTER_MANAGEMENT_KEY"]
        trial_label = f"horizon-hermes-{uuid.uuid4().hex[:8]}"

        # Prefix the user's instruction with a capabilities preamble so
        # hermes knows it has both (a) prior-session memory via
        # `session_search` (the trace was seeded into SessionDB at install
        # time) and (b) the environment-owned tool CLI at
        # /tools/horizon-tools. Without this, hermes's autonomous mode
        # tends to give up after `ls /root/` and never discovers either
        # surface — making it unfair vs API agents whose tool registry is
        # passed natively as LLM function schemas. This tells the agent
        # what's available, not what to do.
        framed_instruction = (
            "You have two key capabilities for this task:\n"
            "  1. `session_search` (your built-in tool): prior-session "
            "history has been seeded into your SessionDB. Search it to "
            "recall context from before this session started.\n"
            "  2. `/tools/horizon-tools` (a CLI on PATH as `horizon-tools`): "
            "the task environment's stateful tools (e.g. sms_list, sms_send, "
            "show_account, document/task ops). Run `horizon-tools --help` "
            "to discover subcommands. These mutate live environment state "
            "and your final reward depends on those mutations.\n\n"
            "Use both as needed. Complete the task and stop when its "
            "success condition is met.\n\n"
            f"Task: {instruction.strip()}"
        )

        t_run_start = time.monotonic()

        async with trial_subkey(
            management_key=management_key,
            label=trial_label,
        ) as tk:
            # Build the env passed to the hermes subprocess. We override
            # OPENROUTER_API_KEY with the per-trial sub-key so every LLM
            # call billed by `hermes chat` lands on the sub-key's ledger.
            run_env = dict(self._resolved_env_vars)
            run_env["OPENROUTER_API_KEY"] = tk.key

            # Run from /workdir so hermes's built-in `terminal`/`search_files`
            # tools see the eval's filesystem (the trace at /workdir/, the env
            # tool CLI at /tools/horizon-tools, state at /state/) instead of
            # the empty /root home dir.
            command = (
                f"cd /workdir && hermes chat -Q -q {shlex.quote(framed_instruction)} "
                f"--provider openrouter --model {shlex.quote(model)} --yolo"
            )
            result = await self.exec_as_root(
                environment,
                command,
                env=run_env,
                timeout_sec=900,
            )
            (self.logs_dir / "hermes-stdout.txt").write_text(result.stdout or "")
            (self.logs_dir / "hermes-stderr.txt").write_text(result.stderr or "")

            # Dump the session we just ran so populate_context_post_run can
            # convert it to an ATIF trajectory.
            dump_cmd = (
                f"cat > /tmp/hermes_dump.py <<'HERMES_DUMP_EOF'\n"
                f"{DUMP_PY}HERMES_DUMP_EOF\n"
                f"{HERMES_VENV_PY} /tmp/hermes_dump.py"
            )
            dump = await self.exec_as_root(environment, dump_cmd, timeout_sec=60)
            (self.logs_dir / "hermes-session.json").write_text(dump.stdout or "{}")

            # Read the per-trial sub-key's accumulated USD spend before it's
            # deleted — this is the `hermes chat` task-run cost (the build cost
            # is tracked separately at install time).
            self._trial_run_cost_usd = await _subkey_usage_usd(tk.key)

        t_run_end = time.monotonic()

        # Stash for populate_context_post_run.
        self._trial_key = tk
        self._trial_timings = {
            "chat": round(t_run_end - t_run_start, 3),
            "total": round(t_run_end - t_run_start, 3),
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        session_path = self.logs_dir / "hermes-session.json"
        stdout_path = self.logs_dir / "hermes-stdout.txt"

        session_data: dict = {}
        if session_path.exists():
            try:
                session_data = json.loads(session_path.read_text() or "{}")
            except json.JSONDecodeError:
                session_data = {}

        messages = session_data.get("messages") or []
        session = session_data.get("session") or {}
        model = session.get("model") or self.model_name or DEFAULT_MODEL

        steps: list[Step] = []
        total_prompt = total_completion = 0

        if messages:
            for idx, msg in enumerate(messages, start=1):
                role = msg.get("role") or "user"
                source = "agent" if role in ("assistant", "tool") else "user"
                content = msg.get("content") or ""
                token_count = int(msg.get("token_count") or 0)
                step_metrics = None
                if source == "agent" and token_count:
                    total_completion += token_count
                    step_metrics = Metrics(completion_tokens=token_count)
                elif source == "user" and token_count:
                    total_prompt += token_count
                    step_metrics = Metrics(prompt_tokens=token_count)
                steps.append(
                    Step(
                        step_id=idx,
                        timestamp=_to_iso(msg.get("timestamp")),
                        source=source,
                        model_name=model if source == "agent" else None,
                        message=str(content),
                        metrics=step_metrics,
                    )
                )
        else:
            # Fallback: no session DB data. Build a minimal 2-step trajectory
            # so the trial still has a valid ATIF artifact.
            final_message = stdout_path.read_text() if stdout_path.exists() else ""
            steps = [
                Step(
                    step_id=1,
                    timestamp=_now_iso(),
                    source="user",
                    message="<task instruction>",
                ),
                Step(
                    step_id=2,
                    timestamp=_now_iso(),
                    source="agent",
                    model_name=model,
                    message=final_message.strip() or "(no output captured)",
                ),
            ]

        trial_key: TrialKeyState | None = getattr(self, "_trial_key", None)
        if trial_key is None:
            trial_key = TrialKeyState(key="")

        # Per-case cost = actual task-run spend + the memory-build cost. The
        # build cost is attributed to EVERY trial of the case (even cache hits)
        # because a standalone Hermes run would always pay it before acting.
        run_cost = float(getattr(self, "_trial_run_cost_usd", 0.0) or 0.0)
        mem_cost = float(getattr(self, "_mem_build_cost_usd", 0.0) or 0.0)
        extra: dict = {
            "cost_usd": trial_key.cost_usd_dict(
                direct_total=round(run_cost + mem_cost, 6),
                breakdown={
                    "task_run": round(run_cost, 6),
                    "memory_build": round(mem_cost, 6),
                },
            ),
            "memory_build_cost_usd": round(mem_cost, 6),
        }
        if getattr(self, "_mem_build_meta", None):
            extra["memory_build"] = self._mem_build_meta
        timings = getattr(self, "_trial_timings", None)
        if timings:
            extra["timing_seconds"] = timings

        trajectory = Trajectory(
            schema_version=ATIF_VERSION,
            session_id=str(session.get("id") or uuid.uuid4()),
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
            extra=extra,
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )
        context.n_input_tokens = total_prompt or None
        context.n_output_tokens = total_completion or None
