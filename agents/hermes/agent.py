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

import json
import os
import shlex
import time
import uuid
from datetime import UTC, datetime

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
ATIF_VERSION = "ATIF-v1.4"


SEED_PY = r'''
"""Translate /workdir/trace.jsonl into one Hermes session per UTC day."""
import json, os, sys, uuid
from collections import defaultdict
from datetime import datetime, timezone

# Hermes ships its state module under /usr/local/lib/hermes-agent.
sys.path.insert(0, "/usr/local/lib/hermes-agent")

from hermes_state import SessionDB  # noqa: E402

trace_path = sys.argv[1]
model = sys.argv[2]

groups = defaultdict(list)
with open(trace_path) as f:
    for raw in f:
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ts = event.get("timestamp", "1970-01-01T00:00:00Z")
        try:
            day = (
                datetime.fromisoformat(ts.replace("Z", "+00:00"))
                .astimezone(timezone.utc)
                .date()
                .isoformat()
            )
        except ValueError:
            day = "undated"
        groups[day].append(event)


def event_to_message(event: dict) -> tuple[str, str] | None:
    data = event.get("message_data") or {}
    etype = data.get("type")
    if etype == "message":
        role = data.get("role") or "user"
        content = data.get("content") or ""
        return (role, str(content))
    if etype == "reasoning":
        return ("assistant", "[reasoning] " + str(data.get("summary") or ""))
    if etype == "function_call":
        name = data.get("name") or "tool"
        args = data.get("arguments") or "{}"
        return ("assistant", f"[tool:{name}] {args}")
    if etype == "function_call_output":
        return ("tool", str(data.get("output") or ""))
    return None


db = SessionDB()
created = 0
for day, events in sorted(groups.items()):
    session_id = f"horizon-seed-{day}-{uuid.uuid4().hex[:8]}"
    db.create_session(session_id=session_id, source="cli", model=model)
    for event in events:
        msg = event_to_message(event)
        if msg is None:
            continue
        role, content = msg
        db.append_message(session_id=session_id, role=role, content=content)
    created += 1
db.close()
print(f"seeded {created} sessions")
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
        config_yaml = (
            "memory:\n"
            "  memory_enabled: true\n"
            "  user_profile_enabled: true\n"
            "model:\n"
            f"  primary: openrouter/{model}\n"
        )
        await self.exec_as_root(
            environment,
            f"mkdir -p {HERMES_HOME} && "
            f"cat > {HERMES_HOME}/config.yaml <<'HERMES_CFG_EOF'\n"
            f"{config_yaml}HERMES_CFG_EOF",
        )

        # Seed past sessions from the task's trace, if present.
        await self.exec_as_root(
            environment,
            (
                f"if [ -f {TRACE_PATH} ]; then "
                f"cat > /tmp/hermes_seed.py <<'HERMES_SEED_EOF'\n"
                f"{SEED_PY}HERMES_SEED_EOF\n"
                f"{HERMES_VENV_PY} /tmp/hermes_seed.py {TRACE_PATH} {shlex.quote(model)}; "
                f"else echo '(no trace to seed)'; fi"
            ),
            timeout_sec=120,
        )

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
        # surface — making it unfair vs API agents like trace_mem0 whose
        # tool registry is passed natively as LLM function schemas.
        # This mirrors the spirit of trace_mem0's system prompt: tells
        # the agent what's available, not what to do.
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
        extra: dict = {"cost_usd": trial_key.cost_usd_dict()}
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
