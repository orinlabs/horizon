# hermes

A Harbor wrapper around [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — the upstream Hermes CLI, installed inside the task container via its own `scripts/install.sh` (the same recipe baked into the official `nousresearch/hermes-agent` Docker image).

## What it demonstrates

- **Agent-specific trace ingestion.** Before running, the agent parses `/workdir/trace.jsonl`, groups events by UTC day, and seeds one past session per day into Hermes's native `SessionDB` (the SQLite + FTS5 store at `~/.hermes/state.db`). Hermes's built-in `session_search` tool can then recall the prior history on its own terms — no manual prompt stuffing.
- **ATIF trajectories.** `populate_context_post_run` reads the messages Hermes persisted for the live session and converts them into ATIF `Step`s so `harbor view` renders the full run and the SFT/RL exporters work.

## Cost

**Cold install**: ~7-10 min per trial — hermes pulls `uv`, Python 3.11, Node.js 22, clones the repo, `npm install`s its deps, fetches a camoufox browser bundle, and builds its Python venv with `[all]` extras.

**Warm install (recommended for iteration)**: ~5 s per trial when the install is cached in a named Docker volume. The agent detects an existing install at `/usr/local/lib/hermes-agent/venv` and skips `install.sh` entirely.

## Run

Single task:

```bash
harbor run \
    -p evals/01-example-catering-vendor \
    --agent-import-path hermes.agent:HermesAgent \
    -m anthropic/claude-sonnet-4.6 \
    --agent-setup-timeout-multiplier 3 \
    --ae OPENROUTER_API_KEY=sk-or-...
```

Full benchmark dataset:

```bash
harbor run \
    -d orinlabs/horizon-1-public \
    --agent-import-path hermes.agent:HermesAgent \
    -m anthropic/claude-sonnet-4.6 \
    --agent-setup-timeout-multiplier 3 \
    --ae OPENROUTER_API_KEY=sk-or-...
```

## Faster iteration via cache mounts (optional)

For repeat runs against the same task, three bind mounts persist hermes's install tree, state DB, and `uv` cache across trials:

```bash
mkdir -p /tmp/horizon-hermes-install /tmp/horizon-hermes-uv /tmp/horizon-hermes-state

harbor run \
    -p evals/01-example-catering-vendor \
    --agent-import-path hermes.agent:HermesAgent \
    -m anthropic/claude-sonnet-4.6 \
    --agent-setup-timeout-multiplier 3 \
    --ae OPENROUTER_API_KEY=sk-or-... \
    --mounts-json '[
        {"type": "bind", "source": "/tmp/horizon-hermes-install", "target": "/usr/local/lib/hermes-agent"},
        {"type": "bind", "source": "/tmp/horizon-hermes-uv",      "target": "/root/.local"},
        {"type": "bind", "source": "/tmp/horizon-hermes-state",   "target": "/root/.hermes"}
    ]'
```

First run against fresh mount dirs: ~10 min. Every subsequent run: ~30 s total (install probe + the actual `hermes chat` call).

To force a clean reinstall: `rm -rf /tmp/horizon-hermes-install /tmp/horizon-hermes-uv /tmp/horizon-hermes-state`.

Note: Harbor's `--mounts-json` emits only a `services.main.volumes` block (no top-level `volumes:` section), so *named* Docker volumes (`"type": "volume"`) won't compose. Bind mounts work because Docker Compose doesn't require declaration for those.

## Alternative: a pre-baked base image

If volumes aren't an option in your environment, publish a base image with hermes pre-installed and point your task at it:

```toml
# evals/your-task/task.toml
[environment]
docker_image = "your-org/ubuntu-hermes:latest"
```

The agent stays the same — its fast-path check sees the existing install and skips.

## Cost tracking

For per-trial USD attribution, set `OPENROUTER_PROVISIONING_KEY` (a key
with `keys:create` permission) alongside your `OPENROUTER_API_KEY`:

```bash
export OPENROUTER_PROVISIONING_KEY=sk-or-prov-...
export OPENROUTER_API_KEY=sk-or-...           # fallback used if provisioning fails
```

The agent mints a disposable OpenRouter sub-key per trial (capped at
$5.00 by default), routes the `hermes chat` subprocess's traffic through
it, snapshots usage at trial end, and deletes the sub-key.
`trajectory.extra.cost_usd.total` is then exact USD spent by the trial,
including subprocess CLI calls.

Without `OPENROUTER_PROVISIONING_KEY`, trials run on the shared
`OPENROUTER_API_KEY` and per-trial cost can't be isolated from
concurrent traffic. `cost_usd.mode = "shared_key"` documents this.

## Architecture notes

`install()` does four things:
1. `apt-get install` the prereqs from Hermes's upstream Dockerfile.
2. `curl | bash` Hermes's `install.sh --skip-setup` (non-interactive mode).
3. Write a minimal `~/.hermes/config.yaml` pinning OpenRouter + the model.
4. Execute an embedded Python script via Hermes's own venv to seed `SessionDB` from `/workdir/trace.jsonl`.

`run()` invokes `hermes chat -Q -q "<instruction>" --provider openrouter --model <model> --yolo` and captures stdout / stderr. Then dumps the most recent CLI session from `state.db` for trajectory conversion.

`populate_context_post_run()` converts the dumped messages into an ATIF `Trajectory` with per-step metrics and writes `trajectory.json`.
