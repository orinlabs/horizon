# Horizon-1

![Horizon-1](docs/benchmark.png)

[![Stars](https://img.shields.io/github/stars/orinlabs/horizon-1?style=flat&logo=github&cacheSeconds=300)](https://github.com/orinlabs/horizon-1/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/orinlabs/horizon-1?cacheSeconds=300)](https://github.com/orinlabs/horizon-1/commits/main)
[![License](https://img.shields.io/github/license/orinlabs/horizon-1?cacheSeconds=300)](./LICENSE)
[![Harbor](https://img.shields.io/badge/harness-harbor-blue)](https://www.harborframework.com/)

A learning benchmark for extremely long-horizon agents, packaged as [Harbor](https://www.harborframework.com/) tasks and agents.

**Read the research post:** [Introducing Horizon](https://www.orinlabs.ai/research/horizon)

## Purpose

As agents get more autonomous, their ability to learn on the job has become a critical bottleneck for usefulness. Existing memory benchmarks ([LoCoMo](https://arxiv.org/abs/2402.17753), [LongMemEval](https://arxiv.org/abs/2410.10813)) measure reactive chatbot applications, not autonomous agents. Existing learning benchmarks like [ARC-AGI](https://arcprize.org/) measure acquisition and application of skills, but use sandboxed environments that are not representative of complex work.

Horizon-1 measures whether an agent can acquire learnings from a long first-person history (see task example below) and apply them later in an environment. It makes no distinction between models and harnesses: the target is the utility of the learning system, regardless of how it is crafted.

## Structure

The bulk of Horizon-1 is private to prevent overfitting. We have included a few example eval cases in this public repo, including a public [HuggingFace dataset](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces) of traces, to show how the benchmark is structured.

Each trace is downloaded into the environment image at `/workdir/trace.jsonl` at build time. The agent is given a chance to ingest this trace however it wants before the task starts. Then, the task starts and the agent must use the trace (or any derived representations of it) to complete the task.

The average trace in the private set is ~30M tokens of data; the public examples are much smaller. Browse the full public set at [orinlabs/horizon-1-example-traces](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces).

The agent is then evaluated on accuracy, speed, and cost.


## Task Example

Imagine that in a past session the agent discovered that `curl` is broken on this machine and switched to `wget`. This would be recorded in the trace as:

```jsonl
{"timestamp": "2026-03-19T09:42:11+00:00", "message_data": {"id": null, "type": "reasoning", "summary": "Need to grab the release tarball. Reaching for curl by default."}}
{"timestamp": "2026-03-19T09:42:12+00:00", "message_data": {"id": "tc_a1", "name": "shell_exec", "type": "function_call", "call_id": "tc_a1", "arguments": "{\"command\": \"curl -fLO https://files.example.com/release-1.4.2.tar.gz\"}"}}
{"timestamp": "2026-03-19T09:42:12+00:00", "message_data": {"type": "function_call_output", "call_id": "tc_a1", "output": "curl: symbol lookup error: undefined symbol: SSL_get1_peer_certificate\nexit code: 127"}}
{"timestamp": "2026-03-19T09:42:14+00:00", "message_data": {"id": null, "type": "reasoning", "summary": "curl is broken on this box — bad libssl link. Falling back to wget."}}
{"timestamp": "2026-03-19T09:42:15+00:00", "message_data": {"id": "tc_a2", "name": "shell_exec", "type": "function_call", "call_id": "tc_a2", "arguments": "{\"command\": \"wget https://files.example.com/release-1.4.2.tar.gz\"}"}}
{"timestamp": "2026-03-19T09:42:18+00:00", "message_data": {"type": "function_call_output", "call_id": "tc_a2", "output": "release-1.4.2.tar.gz  100%[==================>]  18.4M ... saved"}}
```

This part of the trace would be buried in hundreds of days of real, unrelated activity. Today it's asked to download another file — does it remember to reach for `wget`, or does it rediscover the breakage by trying `curl` first?

The agent is expected to remember that `curl` is broken and use wget first this time. Reward is assigned based on whether the agent tries to use `curl` or `wget` first. `curl` is the default choice for most models, so if the agent deviates from this behavior, we assume it has learned something from the trace.


## Installation

Requires [Docker](https://docs.docker.com/get-docker/) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/orinlabs/horizon-1.git
cd horizon-1
uv tool install harbor --with-editable .
```

`--with-editable .` installs this repo's reference agents (e.g. `trace_shell_context`) alongside Harbor so they can be referenced by import path.

## Running the benchmark

Get an [OpenRouter](https://openrouter.ai/keys) API key, then run the full dataset against any agent:

```bash
# built-in Harbor agent (e.g. terminus-2)
harbor run \
  -d orinlabs/horizon-1-public \
  -a terminus-2 \
  -m openrouter/openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...

# reference an agent shipped in this repo
harbor run \
  -d orinlabs/horizon-1-public \
  --agent-import-path trace_shell_context.agent:TraceShellContextAgent \
  -m openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...
```

`-d` runs every task in the [orinlabs/horizon-1-public](https://hub.harborframework.com/datasets/orinlabs/horizon-1-public) dataset. To target a single task instead, swap `-d <name>` for `-p evals/<task-dir>`.

Reference agents live under `agents/`. Any [Harbor built-in agent](https://www.harborframework.com/docs/agents) works with `-a <name>`. Results land in `jobs/<job-name>/` — browse them with `harbor view jobs`.

## Running on Daytona

Trials run on local Docker by default, one at a time. For parallel cloud runs, get a [Daytona](https://www.daytona.io/) API key and pass `-e daytona`:

```bash
export DAYTONA_API_KEY=dtn_...

harbor run \
  -d orinlabs/horizon-1-public \
  -a terminus-2 \
  -m openrouter/openai/gpt-4o-mini \
  -e daytona \
  -n 32 \
  --ae OPENROUTER_API_KEY=sk-or-...
```

## Submitting an agent

We're collecting community submissions for the public leaderboard. The flow is intentionally lightweight — open a PR, then email us.

1. **Drop your agent under `agents/<your-agent-name>/`**, mirroring the layout of an existing reference agent like [`agents/trace_window/`](./agents/trace_window/) or [`agents/trace_shell_context/`](./agents/trace_shell_context/):

   ```
   agents/<your-agent-name>/
   ├── __init__.py     # re-export your agent class
   └── agent.py        # subclass harbor.agents.base.BaseAgent
   ```

2. **Run it against the public set locally** to confirm it loads end-to-end:

   ```bash
   harbor run \
     -d orinlabs/horizon-1-public \
     --agent-import-path <your_agent_name>.agent:<YourAgentClass> \
     -m openai/gpt-4o-mini \
     --ae OPENROUTER_API_KEY=sk-or-...
   ```

3. **Open a PR against [`orinlabs/horizon-1`](https://github.com/orinlabs/horizon-1)** with just the new `agents/<your-agent-name>/` directory. Don't modify reference agents, the eval set, or the dataset manifest.
4. **Email [horizon@orinlabs.ai](mailto:horizon@orinlabs.ai)** with a link to the PR. Include the agent name, the model(s) you've validated against, and any setup notes (extra env vars, model assumptions, etc.). We'll run it on the private set, post results to the leaderboard, and merge the PR.

A few things worth knowing before you write the agent:

- The agent runs on the host, not in the sandbox — see [How agents and environments are split](#how-agents-and-environments-are-split) below. **Use async I/O for every LLM call**; a synchronous SDK call inside `run()` will block the shared event loop and starve every other parallel trial.
- The full trace can be tens of MB. Read it with `await environment.download_file("/workdir/trace.jsonl", …)` (see `agent_utils.read_trace_file` used by the reference agents) instead of `cat`-ing it through `environment.exec`, which truncates at the agent's stdout cap.
- Don't hard-code tool surfaces. The set of available tools varies per task; the trace's `function_call` items are the source of truth for what's installed in `/usr/local/bin` for that environment.

## How agents and environments are split

Each trial has two pieces in two places:

- **The environment** is a sandbox container (Docker locally, or a Daytona/Modal/etc. sandbox in the cloud). It holds the per-trial filesystem state: `/workdir/trace.jsonl`, the per-tool wrappers `horizon-install-tools` placed in `/usr/local/bin`, the `/state` directory the verifier inspects, etc. The sandbox has no internet access (`allow_internet = false`) and runs no agent code — it only executes shell commands the host sends it.
- **The agent** runs in the `harbor` process on the host (your laptop, a CI runner, wherever you invoked `harbor run`). It owns the conversation history and the `OPENROUTER_API_KEY`, calls the LLM, decides what to do, and dispatches each tool call to its sandbox via `environment.exec(...)`.

When you pass `-n N`, harbor allocates `N` isolated sandboxes AND `N` agent coroutines on the host, all sharing one Python event loop and one HTTP client pool. The sandboxes are perfectly isolated from each other; the agent coroutines are not. **Any synchronous LLM SDK call inside an agent's `run()` will block the event loop and starve the other `N-1` coroutines** — even though their sandboxes are completely independent, they can't dispatch their `environment.exec(...)` calls until the loop is freed. Reference agents in `agents/` use `AsyncOpenAI` and `await` every API call for this reason; if you write your own agent, do the same.
