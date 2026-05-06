"""OpenClawLcmAgent — Harbor wrapper around openclaw + lossless-claw.

Active-ingestion baseline alongside trace_mem0 and hermes:

  - mem0  : owns extraction → vector store → semantic recall
  - hermes: agent's own MEMORY.md/USER.md (passive), SessionDB seeded raw
  - openclaw: lossless-claw plugin persists every message, builds a DAG of
              summaries, and exposes lcm_grep / lcm_describe / lcm_expand
              tools so the agent can recall from compacted history

Setup (run once per task image, idempotent):
  1. apt-get install nodejs (Node 22.12+ via NodeSource)
  2. npm install -g openclaw
  3. openclaw plugins install @martian-engineering/lossless-claw
  4. Write ~/.openclaw/.env (OPENROUTER_API_KEY) + openclaw.json (model)
  5. Bootstrap the DB schema with `openclaw doctor`
  6. Seed /workdir/trace.jsonl into lcm.db via seed.mjs (direct SQLite writes
     against lossless-claw's documented schema; engine picks up the rows on
     the first chat turn and lazily summarizes via afterTurn)

Run:
  openclaw agent --message "<framed instruction>" --session-id <SEED_ID> \\
                 --local --json

Per-trial cost is dominated by openclaw inference (claude-opus-4.7-1m by
default) plus lossless-claw's first-turn summarization pass over the
seeded raw history.
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

from agent_utils import (
    TrialKeyState,
    begin_trial_key,
    finalize_trial_key,
)
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
    Step,
    Trajectory,
)


TRACE_PATH = "/workdir/trace.jsonl"
OPENCLAW_HOME = "/root/.openclaw"
DEFAULT_MODEL = "anthropic/claude-opus-4.7"
DEFAULT_SUMMARY_MODEL = "openai/gpt-5-mini"
DEFAULT_EXPANSION_MODEL = "anthropic/claude-opus-4.7"
ATIF_VERSION = "ATIF-v1.4"
SEED_SESSION_PREFIX = "horizon-seed"

# Pinned to lossless-claw's most recent published version (0.9.x line).
# `npm:` prefix bypasses openclaw's clawhub registry resolution (which
# was hitting ECONNRESET in our Daytona sandboxes) and goes straight to
# npm. Functionally identical install path.
LCM_PACKAGE = "npm:@martian-engineering/lossless-claw"

# Path inside the container where we drop the trace converter after
# uploading it from the host. /tmp survives between exec calls but not
# across sandbox restarts; that's fine because install() always re-runs.
CONVERTER_SCRIPT_REMOTE = "/tmp/openclaw_convert_trace.mjs"

# Agent-dir name lcm-tui reads sessions from
# (~/.openclaw/agents/<AGENT_NAME>/sessions/*.jsonl). Has to match the
# `<agent>` arg we pass to `lcm-tui backfill`. We use "main" because
# openclaw's default agent dir is workspace-main and the gateway's
# session keys come back as `agent:main:explicit:<sid>`.
LCM_AGENT_NAME = "main"

# Go release tarball to fetch when building lcm-tui from source. Pinned
# because lossless-claw/tui requires Go >=1.24 (per /tmp/lossless-claw/
# tui/README.md), and Debian 13's golang-go is 1.21.x.
GO_VERSION = "1.24.0"
GO_TARBALL_URL = (
    f"https://go.dev/dl/go{GO_VERSION}.linux-amd64.tar.gz"
)

# Local gateway port. Default 18789; we keep it on loopback so we can
# disable auth (gateway.auth.mode = "none"). The gateway is the long-
# running daemon that the agent CLI talks to over WebSocket; it's the
# only mode where lossless-claw's runtime hooks (assemble + lcm_grep
# tool registration) actually fire. `--local` uses an in-process
# embedded runtime that loads plugin manifests but skips the JS entry
# point that calls api.registerTool / api.registerContextEngine.
GATEWAY_PORT = 18789


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class OpenClawLcmAgent(BaseInstalledAgent):
    """Harbor-native wrapper around openclaw + lossless-claw."""

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
        return "openclaw-lcm"

    def version(self) -> str | None:
        return "0.1.0"

    def get_version_command(self) -> str | None:
        return "openclaw --version 2>&1 | head -1"

    async def install(self, environment: BaseEnvironment) -> None:
        if "OPENROUTER_API_KEY" not in self._resolved_env_vars:
            raise RuntimeError(
                "OpenClawLcmAgent requires OPENROUTER_API_KEY; set it in your "
                "host env before invoking `harbor run`."
            )
        management_key = os.environ["OPENROUTER_MANAGEMENT_KEY"]
        model = self.model_name or DEFAULT_MODEL

        # Sub-key spans both install (lcm-tui backfill summarization) and
        # run (openclaw agent), so we use begin/finalize directly rather
        # than the `trial_subkey` context manager (which can't span two
        # methods).
        trial_key = await begin_trial_key(
            management_key=management_key,
            label=f"horizon-openclaw-lcm-{uuid.uuid4().hex[:8]}",
        )
        self._trial_key = trial_key
        # Build a separate env dict with the sub-key so we don't mutate
        # the base class's _resolved_env_vars (which would leave a dead
        # key after finalize deletes the sub-key).
        self._trial_env = {**self._resolved_env_vars, "OPENROUTER_API_KEY": trial_key.key}
        api_key = trial_key.key
        self._install_started = time.monotonic()

        try:
            await self._do_install(environment, model, api_key)
        except BaseException:
            # If install fails after minting the sub-key, finalize it now
            # so we don't leak an orphaned key on OpenRouter.
            await finalize_trial_key(trial_key)
            raise

        self._install_finished = time.monotonic()

    async def _do_install(
        self,
        environment: BaseEnvironment,
        model: str,
        api_key: str,
    ) -> None:
        # Skip the apt + npm install if openclaw is already on PATH (warm
        # cache from a previous trial in the same sandbox).
        probe = await environment.exec(
            "command -v openclaw && command -v node",
            user="root",
        )
        already_installed = probe.return_code == 0

        if not already_installed:
            # Node 22.12+ is required by openclaw (hard check at startup).
            # Debian's bundled node is too old, so use NodeSource's setup_22.x.
            await self.exec_as_root(
                environment,
                (
                    "apt-get update && "
                    "apt-get install -y --no-install-recommends "
                    "curl ca-certificates git build-essential python3 && "
                    "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
                    "apt-get install -y --no-install-recommends nodejs"
                ),
                timeout_sec=600,
            )
            # Pin openclaw to 2026.4.29 — the last 2026.4 release.
            # 2026.5.x silently no-ops registerContextEngine for plugins
            # whose manifest doesn't declare contracts.contextEngines.
            # Lossless-claw 0.9.3 only ships contracts.tools, so on 2026.5.x
            # its register() runs but lcm_grep et al never reach the agent
            # tool list (open issue Martian-Engineering/lossless-claw#588).
            # 2026.4.27 restored the registerContextEngine root-alias bridge
            # and 2026.4.29 inherits that fix without the 2026.5.2 strict-
            # acceptance regression.
            await self.exec_as_root(
                environment,
                "npm install -g openclaw@2026.4.29 better-sqlite3",
                timeout_sec=600,
            )

        # Step 1: write provider creds + a *minimal* config (just agents
        # so openclaw can find a default agent). We DON'T write our
        # plugins config yet — `openclaw plugins install` mutates the
        # config file to record `plugins.installs.<id>.installPath`,
        # and overwriting after install would erase that record (which
        # makes the plugin invisible at runtime even though npm shows
        # it installed).
        await self.exec_as_root(
            environment,
            f"mkdir -p {OPENCLAW_HOME} && "
            f"cat > {OPENCLAW_HOME}/.env <<'OC_ENV_EOF'\n"
            f"OPENROUTER_API_KEY={api_key}\n"
            "OC_ENV_EOF",
        )

        base_config = json.dumps(
            {
                "agents": {
                    # `skipBootstrap` suppresses openclaw's "you just
                    # woke up, introduce yourself" BOOTSTRAP.md flow
                    # that would otherwise hijack the first turn.
                    # (`skipOptionalBootstrapFiles` exists only in
                    # 2026.5.x; we're pinned to 2026.4.29 because of
                    # lossless-claw#588, so omit that key.)
                    "defaults": {
                        "skipBootstrap": True,
                    },
                    "list": [
                        {
                            # "main" matches openclaw's default agent dir
                            # (workspace-main) and the sessionKey scheme
                            # (`agent:main:explicit:<sid>`), and is the
                            # <agent> arg we pass to `lcm-tui backfill`.
                            "id": LCM_AGENT_NAME,
                            "default": True,
                            "model": f"openrouter/{model}",
                        }
                    ],
                },
            },
            indent=2,
        )
        await self.exec_as_root(
            environment,
            f"cat > {OPENCLAW_HOME}/openclaw.json <<'OC_CFG_EOF'\n"
            f"{base_config}\n"
            "OC_CFG_EOF",
        )

        # Step 2: install lossless-claw. This appends
        # plugins.installs.<id> + plugins.entries.<id> to openclaw.json
        # in place — we MUST run after writing the base config, not
        # before, or our overwrite would clobber the install record.
        # Retry: clawhub fetches occasionally hit ECONNRESET on cold
        # sandboxes (saw this on 2026.4.29 install path).
        if not already_installed:
            await self.exec_as_root(
                environment,
                (
                    f"for i in 1 2 3; do "
                    f"  openclaw plugins install {shlex.quote(LCM_PACKAGE)} && exit 0; "
                    f"  echo \"plugin install attempt $i failed, retrying...\"; "
                    f"  sleep 5; "
                    f"done; "
                    f"echo 'plugin install failed after 3 attempts'; exit 1"
                ),
                timeout_sec=600,
            )

        # Step 3: patch the post-install config to (a) bind
        # contextEngine slot to lossless-claw, (b) inject our
        # summary/expansion model overrides, and (c) set gateway auth
        # to "none" so we can talk to the local-loopback daemon
        # without seeding a shared secret. Done via Node mutator so
        # the merge stays JSON-aware (preserves plugins.installs).
        patch_js = (
            "const fs = require('node:fs');"
            "const p = '" + OPENCLAW_HOME + "/openclaw.json';"
            "const c = JSON.parse(fs.readFileSync(p, 'utf8'));"
            "c.plugins = c.plugins || {};"
            "c.plugins.slots = c.plugins.slots || {};"
            "c.plugins.slots.contextEngine = 'lossless-claw';"
            "c.plugins.entries = c.plugins.entries || {};"
            "c.plugins.entries['lossless-claw'] = c.plugins.entries['lossless-claw'] || {};"
            "c.plugins.entries['lossless-claw'].enabled = true;"
            "c.plugins.entries['lossless-claw'].config = Object.assign("
            "  {}, c.plugins.entries['lossless-claw'].config || {},"
            "  {"
            f"    summaryModel: 'openrouter/{DEFAULT_SUMMARY_MODEL}',"
            f"    expansionModel: 'openrouter/{DEFAULT_EXPANSION_MODEL}',"
            "    leafChunkTokens: 8000,"
            "    leafMinFanout: 4"
            "  });"
            "c.gateway = c.gateway || {};"
            "c.gateway.mode = 'local';"
            f"c.gateway.port = {GATEWAY_PORT};"
            # Force loopback. In container envs the gateway auto-binds
            # to 0.0.0.0 and then refuses to start with auth.mode=none
            # because that would publicly expose it. We're co-located
            # with the agent CLI in the same sandbox, so 127.0.0.1
            # is fine.
            "c.gateway.bind = 'loopback';"
            "c.gateway.auth = c.gateway.auth || {};"
            "c.gateway.auth.mode = 'none';"
            "fs.writeFileSync(p, JSON.stringify(c, null, 2));"
            "console.log('plugin entries:', Object.keys(c.plugins.entries || {}));"
            "console.log('installs:', Object.keys(c.plugins.installs || {}));"
        )
        await self.exec_as_root(
            environment,
            f"node -e {shlex.quote(patch_js)}",
            timeout_sec=60,
        )

        # Step 4: install lcm-tui (lossless-claw's Go-based maintenance
        # binary). It owns the proper ingestion path: per-session import
        # PLUS depth-aware compaction (leaf summaries → condensed
        # roll-ups → DAG), driven by the configured summarization model.
        # Without this step we'd be back to direct-SQL inserts and
        # missing the actual lossless-claw pipeline that distinguishes
        # this engine from raw text search.
        if not already_installed:
            await self.exec_as_root(
                environment,
                (
                    f"curl -fsSL {GO_TARBALL_URL} | "
                    "tar -C /usr/local -xz && "
                    "echo 'export PATH=/usr/local/go/bin:/root/go/bin:$PATH' "
                    ">>/etc/profile.d/lcm-go.sh"
                ),
                timeout_sec=300,
            )
            await self.exec_as_root(
                environment,
                (
                    "PATH=/usr/local/go/bin:$PATH "
                    "go install github.com/Martian-Engineering/lossless-claw/tui@latest "
                    "&& mv /root/go/bin/tui /usr/local/bin/lcm-tui"
                ),
                timeout_sec=600,
            )

        # Step 5: convert /workdir/trace.jsonl into per-day JSONL
        # session files at ~/.openclaw/agents/main/sessions/. Each day
        # becomes its own backfill input file.
        converter_local = Path(__file__).parent / "convert_trace.mjs"
        await environment.upload_file(
            str(converter_local), CONVERTER_SCRIPT_REMOTE
        )
        await self.exec_as_root(environment, f"chmod +x {CONVERTER_SCRIPT_REMOTE}")

        seed_session = f"{SEED_SESSION_PREFIX}-{uuid.uuid4().hex[:8]}"
        convert_result = await self.exec_as_root(
            environment,
            (
                f"if [ ! -f {TRACE_PATH} ]; then "
                "  echo '{\"sessions\":[],\"skipped\":\"no trace\"}'; "
                "  exit 0; "
                "fi; "
                f"node {CONVERTER_SCRIPT_REMOTE} {TRACE_PATH} "
                f"{LCM_AGENT_NAME} {shlex.quote(seed_session)}"
            ),
            timeout_sec=120,
        )
        (self.logs_dir / "convert-stdout.txt").write_text(convert_result.stdout or "")

        # Step 6: run `lcm-tui backfill` per day. Each invocation imports
        # that day's messages into lcm.db AND runs leaf+condensed
        # compaction with the configured summary model. Cost: ~5-15
        # gpt-5-mini calls per backfill, ~$0.05-0.20/day.
        # We pass the openrouter base-url + provider=openai (OpenAI-
        # compat) and OPENAI_API_KEY. lcm-tui's anthropicClient handles
        # both providers via the --provider flag; openrouter is reached
        # via the openai code path with a custom base-url.
        await self.exec_as_root(
            environment,
            (
                "set -e; "
                # Parse `convert_trace.mjs` output (the JSON line we
                # captured above) and loop over its session ids.
                "SESSIONS_JSON=$(cat <<'CONVOUT_EOF'\n"
                f"{convert_result.stdout or '{}'}\n"
                "CONVOUT_EOF\n"
                "); "
                "echo \"$SESSIONS_JSON\" | "
                f"OPENAI_API_KEY={shlex.quote(api_key)} "
                "node -e '"
                "let raw = \"\"; "
                "process.stdin.on(\"data\", d => raw += d); "
                "process.stdin.on(\"end\", () => {"
                "  let parsed = {}; try { parsed = JSON.parse(raw); } catch {} "
                "  const sessions = (parsed.sessions || []).map(s => s.sessionId); "
                "  console.log(sessions.join(\"\\n\")); "
                "});' > /tmp/lcm-sessions.txt; "
                "echo 'sessions to backfill:'; cat /tmp/lcm-sessions.txt; "
                "TOTAL=$(wc -l < /tmp/lcm-sessions.txt); "
                "INDEX=0; "
                "while IFS= read -r SID; do "
                "  [ -z \"$SID\" ] && continue; "
                "  INDEX=$((INDEX+1)); "
                "  echo \"[backfill $INDEX/$TOTAL] $SID\"; "
                f"  OPENAI_API_KEY={shlex.quote(api_key)} "
                f"  LCM_TUI_SUMMARY_PROVIDER=openai "
                f"  LCM_TUI_SUMMARY_MODEL={shlex.quote('openrouter/' + DEFAULT_SUMMARY_MODEL)} "
                f"  LCM_TUI_SUMMARY_BASE_URL=https://openrouter.ai/api/v1 "
                f"  lcm-tui backfill {LCM_AGENT_NAME} \"$SID\" "
                f"     --apply --provider openai --model {shlex.quote('openrouter/' + DEFAULT_SUMMARY_MODEL)} "
                f"     --base-url https://openrouter.ai/api/v1 "
                "     2>&1 | tail -20 || echo 'WARN: backfill failed for '\"$SID\"; "
                "done < /tmp/lcm-sessions.txt"
            ),
            env=self._trial_env,
            timeout_sec=1800,
        )
        # Stash the session id on self so run() picks the same one.
        self._seed_session_id = seed_session

        # Diagnostic: dump plugin runtime state to /tmp/oc-diag.txt so
        # we can inspect it after the trial. Harbor's trial.log only
        # shows "Command outputs captured" without the stdout, so we
        # need the file to read it back in run().
        await self.exec_as_root(
            environment,
            (
                "{ "
                "echo '--- openclaw --version ---'; "
                "openclaw --version 2>&1 || true; "
                "echo; echo '--- npm list -g --depth=0 ---'; "
                "npm list -g --depth=0 2>&1 || true; "
                "echo; echo '--- openclaw plugins list ---'; "
                "openclaw plugins list 2>&1 || true; "
                "echo; echo '--- openclaw plugins list --json ---'; "
                "openclaw plugins list --json 2>&1 || true; "
                "echo; echo '--- openclaw plugins inspect lossless-claw ---'; "
                "openclaw plugins inspect lossless-claw 2>&1 || true; "
                "echo; echo '--- ls /usr/lib/node_modules/@martian-engineering ---'; "
                "ls -la /usr/lib/node_modules/@martian-engineering/ 2>&1 || true; "
                "echo; echo '--- openclaw config dump ---'; "
                f"cat {OPENCLAW_HOME}/openclaw.json 2>&1 || true; "
                "} > /tmp/oc-diag.txt 2>&1 || true; "
                "wc -l /tmp/oc-diag.txt"
            ),
            timeout_sec=60,
        )

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = self.model_name or DEFAULT_MODEL
        seed_session = getattr(self, "_seed_session_id", None) or (
            f"{SEED_SESSION_PREFIX}-{uuid.uuid4().hex[:8]}"
        )

        t_run_start = time.monotonic()
        try:
            framed = (
                "You have two key capabilities for this task:\n"
                "  1. LCM tools (`lcm_grep`, `lcm_describe`, `lcm_expand_query`): "
                "your prior chat history has been ingested into the lossless-claw "
                "context engine, organized as one conversation per UTC day. "
                "**IMPORTANT**: pass `allConversations: true` to `lcm_grep` so it "
                "searches across every prior day rather than just the current "
                "(empty) session. Without this flag the search returns nothing "
                "because the seeded conversations live under different "
                "session_ids.\n"
                "  2. `/tools/horizon-tools` (CLI on PATH as `horizon-tools`): "
                "the task environment's stateful tools (sms, account, document, "
                "task ops). Run `horizon-tools --help` to discover subcommands. "
                "Your final reward depends on the live state mutations these "
                "calls produce.\n\n"
                "Use both as needed. Complete the task and stop when its "
                "success condition is met.\n\n"
                f"Task: {instruction.strip()}"
            )

            # --local runs openclaw embedded (no gateway daemon). Gateway
            # mode didn't actually help lossless-claw register tools.
            command = (
                "cd /workdir && "
                f"openclaw agent --message {shlex.quote(framed)} "
                f"--session-id {shlex.quote(seed_session)} "
                f"--model {shlex.quote(f'openrouter/{model}')} "
                "--local --json"
            )
            result = await self.exec_as_root(
                environment,
                command,
                env=self._trial_env,
                timeout_sec=1200,
            )
            (self.logs_dir / "openclaw-stdout.txt").write_text(result.stdout or "")
            (self.logs_dir / "openclaw-stderr.txt").write_text(result.stderr or "")
        finally:
            # finalize_trial_key snapshots the sub-key + deletes it
            # AND runs the overlap awaitables (sandbox-side stats/diag
            # captures) concurrently with the activity-ledger settle so
            # they cost no extra wall time. The finally also covers
            # crash paths so a mid-run exception still snapshots cost.
            await finalize_trial_key(
                self._trial_key,
                overlap=[
                    self._capture_lcm_stats(environment),
                    self._capture_oc_diag(environment),
                ],
            )

            t_run_end = time.monotonic()
            install_started = getattr(self, "_install_started", t_run_start)
            install_finished = getattr(self, "_install_finished", t_run_start)
            self._trial_timings = {
                "install": round(install_finished - install_started, 3),
                "chat": round(t_run_end - t_run_start, 3),
                "total": round(t_run_end - install_started, 3),
            }

    async def _capture_lcm_stats(self, environment: BaseEnvironment) -> None:
        """Snapshot lcm.db row counts so we can audit how much got summarized."""
        result = await environment.exec(
            "sqlite3 ~/.openclaw/lcm.db "
            "\"SELECT 'conversations:'||COUNT(*) FROM conversations; "
            "SELECT 'messages:'||COUNT(*) FROM messages; "
            "SELECT 'summaries:'||COUNT(*) FROM summaries;\" "
            "2>/dev/null || true",
            user="root",
        )
        (self.logs_dir / "lcm-stats.txt").write_text(result.stdout or "")

    async def _capture_oc_diag(self, environment: BaseEnvironment) -> None:
        """Pull the install-time diagnostic; harbor's trial.log doesn't keep stdout."""
        result = await environment.exec(
            "cat /tmp/oc-diag.txt 2>/dev/null || true", user="root"
        )
        (self.logs_dir / "oc-diag.txt").write_text(result.stdout or "")

    def populate_context_post_run(self, context: AgentContext) -> None:
        stdout_path = self.logs_dir / "openclaw-stdout.txt"
        model = self.model_name or DEFAULT_MODEL

        final_message = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        session_id = str(uuid.uuid4())

        if stdout_path.exists():
            raw = stdout_path.read_text()
            # openclaw --json emits one JSON object per agent turn (NDJSON);
            # the last one carries the final message + usage totals.
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    if "message" in obj:
                        final_message = str(obj.get("message") or "")
                    if "usage" in obj and isinstance(obj["usage"], dict):
                        usage["prompt_tokens"] = int(
                            obj["usage"].get("prompt_tokens") or 0
                        )
                        usage["completion_tokens"] = int(
                            obj["usage"].get("completion_tokens") or 0
                        )
                    if "sessionId" in obj:
                        session_id = str(obj["sessionId"])
            if not final_message:
                final_message = raw.strip()[-4000:]

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
                message=final_message or "(no output captured)",
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
            session_id=session_id,
            agent=Agent(
                name=self.name(),
                version=self.version() or "unknown",
                model_name=model,
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=usage["prompt_tokens"],
                total_completion_tokens=usage["completion_tokens"],
                total_steps=len(steps),
            ),
            extra=extra,
        )
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory.to_json_dict(), indent=2)
        )
        context.n_input_tokens = usage["prompt_tokens"] or None
        context.n_output_tokens = usage["completion_tokens"] or None
