# Horizon-1 agents

Horizon measures whether an agent can **learn from a long prior-session trace** and apply that knowledge in a live task environment. Agents fall into three groups:

1. **Harbor built-in agents** — Claude Code and Codex ship with [Harbor](https://www.harborframework.com/docs/agents) and run via `-a` / `--agent`. No code in this directory.
2. **Reference agents in this repo** — custom `BaseAgent` / `BaseInstalledAgent` implementations used on the [Horizon leaderboard](https://www.orinlabs.ai/research/horizon) or for benchmark integrity checks.
3. **Shared utilities** — `agent_utils.py` and prompt templates used across the above.

Every trial has two pieces: the **agent** runs on the host (in the `harbor` process), and the **environment** is an isolated sandbox with `/workdir/trace.jsonl`, task tools, and verifier state. Read the root [README](../README.md#how-agents-and-environments-are-split) for the full split.

## Harbor built-in: Claude Code and Codex

Harbor pre-integrates popular CLI agents as **installed agents**: it installs the tool inside the sandbox, runs it headless, and records ATIF trajectories. You do not need `--agent-import-path` for these — just pass `-a claude-code` or `-a codex`.

Run `harbor run --help` for the full agent list (`terminus-2`, `openhands`, `gemini-cli`, etc.). On the Horizon leaderboard we benchmark two of them alongside the reference agents below.

### Claude Code (`claude-code`)

Anthropic's CLI coding agent. Harbor installs `claude` in the container and invokes it with the task `instruction.md` as the user message.

```bash
export ANTHROPIC_API_KEY=sk-ant-...

harbor run \
  -d orinlabs/horizon-1-public \
  -a claude-code \
  -m anthropic/claude-opus-4.8 \
  --ak prompt_template_path=agents/claude_code_horizon_prompt.j2
```

**Horizon-specific prompt.** Raw task instructions do not tell Claude Code that `/workdir/trace.jsonl` is its long-horizon memory or how to use the task's CLI tools. For leaderboard runs we wrap the instruction with [`claude_code_horizon_prompt.j2`](./claude_code_horizon_prompt.j2) via Harbor's `prompt_template_path` kwarg (`--ak`). The template frames the trace as prior-session memory and points the agent at `/.horizon/tools/tools.json` for the live tool surface.

Other useful kwargs (see Harbor docs): `reasoning_effort`, `max_turns`, `append_system_prompt`. Pass with `--ak key=value`.

### Codex (`codex`)

OpenAI's Codex CLI agent. Harbor installs `codex` and runs it against the task instruction with the model from `-m`.

```bash
export OPENAI_API_KEY=sk-...

harbor run \
  -d orinlabs/horizon-1-public \
  -a codex \
  -m openai/gpt-5-codex \
  --ak reasoning_effort=high
```

Codex resolves auth from `OPENAI_API_KEY` by default, or from a local `~/.codex/auth.json` when `CODEX_FORCE_AUTH_JSON=1` is set. Like Claude Code, it discovers the sandbox filesystem on its own — there is no separate trace-ingestion phase unless you add a custom prompt template.

## Reference agents (this directory)

These ship with the `horizon-1-agents` package (`uv tool install harbor --with-editable .`). Run them with `--agent-import-path <module>.agent:<Class>`.

| Directory | Harness name | Trace strategy | How it acts on the task |
|---|---|---|---|
| [`openclaw_lcm/`](./openclaw_lcm/) | OpenClaw (LCM) | Per-day backfill into lossless-claw's summary DAG via `lcm-tui` | `openclaw agent` with `lcm_grep` / `lcm_describe` / `lcm_expand_query` |
| [`trace_rlm/`](./trace_rlm/) | RLM | Full trace in a host-side Python REPL; root LM peeks/greps/maps with depth-1 `recurse()` sub-calls | `repl_exec` + `shell_exec` |
| [`trace_rag/`](./trace_rag/) | RAG | Embed trace chunks by UTC day; delete raw trace from sandbox | `trace_search` + `shell_exec` |
| [`hermes/`](./hermes/) | Hermes | Seed trace into Hermes `SessionDB` by UTC day at install time | `hermes chat` with native `session_search` + env tools |
| [`perfect_context/`](./perfect_context/) | PerfectContext *(integrity)* | Hand exact trace line ranges from `trace_pointer.json` into context | Task tools only (no shell, no trace search) |
| [`tools_only/`](./tools_only/) | EnvironmentOnly *(integrity)* | Never reads the trace | Task tools only |

Per-agent READMEs with timing/cost notes: [`openclaw_lcm`](./openclaw_lcm/README.md), [`hermes`](./hermes/README.md), [`trace_rag`](./trace_rag/README.md), [`trace_rlm`](./trace_rlm/README.md).

### OpenClaw (LCM) — `openclaw_lcm`

Wraps [openclaw](https://github.com/openclaw/openclaw) + [lossless-claw](https://github.com/Martian-Engineering/lossless-claw). **Active ingestion**: converts the trace to per-day session files, runs `lcm-tui backfill` to build a summary DAG, then chats with retrieval tools that search across all seeded days. Dominant cost is backfill summarization + inference. See [`openclaw_lcm/README.md`](./openclaw_lcm/README.md) for pinned versions and timeout multipliers.

### RLM — `trace_rlm`

Implements [Recursive Language Models](https://arxiv.org/abs/2512.24601). The trace never enters the root model's context; instead it lives in a persistent REPL with `llm()` / `recurse()` helpers for cheap sub-model calls over slices. The root model writes code to explore the trace, then uses `shell_exec` to complete the task. Recursive model defaults to `openai/gpt-5-mini` (`RLM_SUB_MODEL`).

### RAG — `trace_rag`

Chunks the trace by UTC day, embeds with `openai/text-embedding-3-small` via OpenRouter, and exposes `trace_search(query, k)`. After ingest the raw trace is removed from the sandbox so recall must go through search. Acts via `shell_exec` against the same CLI tools the trace's `function_call` events used.

### Hermes — `hermes`

Wraps [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). At install time it groups the trace by UTC day and seeds Hermes's SQLite `SessionDB`; at run time `session_search` recalls prior history on Hermes's terms. Cold install is slow (~7–10 min); use bind-mount caches documented in [`hermes/README.md`](./hermes/README.md) for iteration.

### Integrity baselines

Not leaderboard harnesses — they validate that tasks are fair:

- **`perfect_context`** — retrieval **ceiling**. Loads verbatim trace slices (line ranges from per-eval `trace_pointer.json`) into context, then must still explore the live environment and invoke task tools. A failure here is reasoning/acting, not retrieval.
- **`tools_only`** — memory **floor**. Only sees current tool outputs; provably cannot read `/workdir/trace.jsonl`. Should score ~0% on tasks that genuinely require long-horizon recall.

Oracle and Anti-Oracle (deterministic solve / no-op scripts) live under each eval's `solution/` directory, not here.

## Shared utilities

[`agent_utils.py`](./agent_utils.py) holds behavior shared across reference agents:

- **`read_trace_file`** — download large traces via `environment.download_file()` instead of `exec cat` (avoids stdout truncation on multi-MB files).
- **`load_environment_tools` / `HorizonToolRegistry`** — parse `/.horizon/tools/tools.json` and dispatch function calls to sandbox CLI wrappers.
- **`trial_subkey`** — mint a disposable, USD-capped OpenRouter sub-key per trial (`OPENROUTER_MANAGEMENT_KEY` required).
- **`usage_cost`** — read per-call USD from OpenRouter responses.

## Quick examples

```bash
# Harbor built-in (Claude Code + Horizon prompt)
harbor run -p evals/01-example-catering-vendor \
  -a claude-code \
  -m anthropic/claude-sonnet-4.6 \
  --ak prompt_template_path=agents/claude_code_horizon_prompt.j2

# Harbor built-in (Codex)
harbor run -p evals/01-example-catering-vendor \
  -a codex \
  -m openai/gpt-5-codex

# Reference agent (RAG)
harbor run -d orinlabs/horizon-1-public \
  --agent-import-path trace_rag.agent:TraceRagAgent \
  -m openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...

# Integrity probe
harbor run -p evals/01-example-catering-vendor \
  --agent-import-path tools_only.agent:ToolsOnlyAgent \
  -m openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...
```

Most reference agents require `OPENROUTER_API_KEY`; several also need `OPENROUTER_MANAGEMENT_KEY` for per-trial cost isolation. Installed agents use their native provider keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.).

## Adding your own agent

See [Submitting an agent](../README.md#submitting-an-agent) in the root README. Mirror an existing directory layout, subclass `harbor.agents.base.BaseAgent`, use async LLM calls, and read traces with `read_trace_file`.
