# openclaw_lcm

A Harbor wrapper around [openclaw](https://github.com/openclaw/openclaw) + the [`@martian-engineering/lossless-claw`](https://github.com/Martian-Engineering/lossless-claw) plugin, exercising lossless-claw's full pipeline: per-day trace ingestion via `lcm-tui backfill` → depth-aware compaction (leaf summaries → condensed roll-ups → DAG) → `lcm_grep` / `lcm_describe` / `lcm_expand_query` retrieval at chat time.

## What it demonstrates

A "memory framework" baseline for long-horizon recall. openclaw_lcm exercises lossless-claw's **summary DAG**:

1. `convert_trace.mjs` groups `/workdir/trace.jsonl` into per-day session files at `~/.openclaw/agents/main/sessions/<day>.jsonl`.
2. `lcm-tui backfill main <day> --apply` imports each day's messages AND runs leaf+condensed compaction with the configured summary model (`gpt-5-mini` by default). Each backfill produces ~5-15 LLM summarization calls, building the DAG.
3. The agent runs via `openclaw agent --message "..." --session-id <X> --local --json`. Lossless-claw's tools surface as native function schemas:
   - `lcm_grep` — regex / FTS5 over messages and summaries (pass `allConversations: true` to search across every seeded day, since each day is a separate conversation)
   - `lcm_describe` — fetch a specific summary or message by id with full lineage
   - `lcm_expand_query` — sub-agent that drills into summaries to recover source detail under a token budget

## Run

Single task:

```bash
harbor run \
    -p evals/01-example-catering-vendor \
    --agent-import-path openclaw_lcm.agent:OpenClawLcmAgent \
    -m anthropic/claude-opus-4.7 \
    --agent-setup-timeout-multiplier 4 \
    --agent-timeout-multiplier 16 \
    --ae OPENROUTER_API_KEY=sk-or-...
```

Full benchmark dataset:

```bash
harbor run \
    -d orinlabs/horizon-public \
    --agent-import-path openclaw_lcm.agent:OpenClawLcmAgent \
    -m anthropic/claude-opus-4.7 \
    --agent-setup-timeout-multiplier 4 \
    --agent-timeout-multiplier 16 \
    --ae OPENROUTER_API_KEY=sk-or-...
```

## Cost & timing

**Cold install** (~3-5 min): apt installs Node 22 (NodeSource) + Go 1.24 (tarball). `npm install -g openclaw@2026.4.29 better-sqlite3`. `openclaw plugins install npm:@martian-engineering/lossless-claw`. `go install github.com/Martian-Engineering/lossless-claw/tui@latest`.

**Backfill phase** (~3-5 min, ~$0.20-0.50 per trial): one `lcm-tui backfill` invocation per UTC day in the trace. gpt-5-mini summarizes each day's chunk, builds DAG roll-ups.

**Chat phase** (~3-5 min, ~$0.50-2 per trial): the agent's actual eval turn with claude-opus-4.7. Uses `lcm_grep` / `lcm_describe` / `lcm_expand` to retrieve from the summary DAG.

Total: ~8-12 min wall clock and ~$1-2 per trial.

## Cost tracking

The agent mints a disposable OpenRouter sub-key once per trial from
`OPENROUTER_MANAGEMENT_KEY` (capped at `OPENROUTER_TRIAL_LIMIT_USD` in
`agents/agent_utils.py`, currently $20.00) and uses it for BOTH phases:
`lcm-tui backfill` gpt-5-mini summarization in `install()` AND the
`openclaw agent` inference loop in `run()`. The sub-key's cumulative
usage is snapshotted at trial end and the key is deleted, so
`trajectory.extra.cost_usd.total` reflects the trial's exact LLM spend
across backfill + chat. If a trial exceeds the cap, OpenRouter returns
HTTP 402 and the agent surfaces it as a normal LLM error.

## Pinned versions

| What | Version | Why |
|---|---|---|
| `openclaw` | `2026.4.29` | 2026.5.x silently no-ops `registerContextEngine` for plugins whose manifest lacks `contracts.contextEngines` (open issue [Martian-Engineering/lossless-claw#588](https://github.com/Martian-Engineering/lossless-claw/issues/588)). 2026.4.29 has the `registerContextEngine` root-alias bridge from 2026.4.27 without the strict-acceptance regression. |
| `lossless-claw` | `0.9.3` (latest) | Ships `contracts.tools` declaration needed by the same loader gate. |
| Go | `1.24.0` | `lcm-tui` requires `>=1.24`. Debian's bundled `golang-go` is too old. |
| Node | `22.x` (NodeSource) | openclaw requires `>=22.12`. |

## Architecture notes

`install()` (per trial):
1. apt prereqs + Node 22 (NodeSource).
2. `npm install -g openclaw@2026.4.29 better-sqlite3`.
3. Write `~/.openclaw/openclaw.json` baseline.
4. `openclaw plugins install npm:@martian-engineering/lossless-claw` (the `npm:` prefix bypasses openclaw's clawhub registry resolution).
5. Patch the post-install config: bind `plugins.slots.contextEngine = lossless-claw`, set summary/expansion model overrides (gpt-5-mini / opus-4.7), set `gateway.bind = "loopback"` and `gateway.auth.mode = "none"`.
6. Download Go 1.24 tarball + `go install ...lossless-claw/tui@latest` → `lcm-tui` binary.
7. Upload `convert_trace.mjs` and run it to write per-day JSONL files.
8. Loop `lcm-tui backfill main <day> --apply --provider openai --model openrouter/openai/gpt-5-mini --base-url https://openrouter.ai/api/v1` over each day. Each invocation imports messages and runs depth-aware compaction.

`run()`: `openclaw agent --message "<framed instruction>" --session-id <X> --local --json`. The framed instruction tells the agent about the LCM tools and reminds it to pass `allConversations: true` to `lcm_grep` (since the seeded data lives across many day-conversations rather than the agent's current empty session).

`populate_context_post_run()`: parses openclaw's NDJSON output, extracts the final assistant message + usage totals, writes an ATIF trajectory.

## Caveats

- **Per-day sessions, not one mega-conversation.** Each UTC day becomes its own lossless-claw conversation. The agent must pass `allConversations: true` to `lcm_grep` to search across all of them. The framed instruction tells the agent this explicitly.
- **Backfill is the cost driver.** Wall clock and LLM spend scale with trace length (more days → more `lcm-tui backfill` invocations). For traces shorter than ~5 days the extra setup time outweighs the retrieval benefit; longer traces are where this agent earns its keep.
- **Pinned to 2026.4.29.** When lossless-claw#588 is resolved (manifest-side fix shipping with a 0.9.4 release, or upstream loosens the loader gate), this pin should be removed and the agent retested on the latest openclaw.
