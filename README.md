# Horizon-1

![Horizon-1](docs/benchmark.png)

[![Stars](https://img.shields.io/github/stars/orinlabs/horizon-1?style=flat&logo=github&cacheSeconds=300)](https://github.com/orinlabs/horizon-1/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/orinlabs/horizon-1?cacheSeconds=300)](https://github.com/orinlabs/horizon-1/commits/main)
[![License](https://img.shields.io/github/license/orinlabs/horizon-1?cacheSeconds=300)](./LICENSE)
[![Harbor](https://img.shields.io/badge/harness-harbor-blue)](https://www.harborframework.com/)

A learning benchmark for extremely long-horizon agents, packaged as [Harbor](https://www.harborframework.com/) tasks and agents.

## Purpose

As agents get more autonomous, their ability to learn on the job has become a critical bottleneck for usefulness. Existing memory benchmarks ([LoCoMo](https://arxiv.org/abs/2402.17753), [LongMemEval](https://arxiv.org/abs/2410.10813)) measure reactive chatbot applications, not autonomous agents. Existing learning benchmarks like [ARC-AGI](https://arcprize.org/) measure acquisition and application of skills, but use sandboxed environments that are not representative of complex work.

Horizon-1 measures whether an agent can acquire learnings from a long first-person history and apply them later in an environment. It makes no distinction between models and harnesses: the target is the utility of the learning system, regardless of how it is crafted.

## Structure

The bulk of Horizon-1 is private to prevent overfitting. We have included a few example eval cases in this public repo, including a public [HuggingFace dataset](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces) of traces, to show how the benchmark is structured.

Each trace is downloaded into the environment image at build time. Each agent is given a chance to ingest this trace however it wants before the task starts. Then, the task starts and the agent must use the trace (or any derived representations of it) to complete the task.

The agent is then evaluated on accuracy, speed, and cost.


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

`-d` runs every task in the `[orinlabs/horizon-1-public](https://hub.harborframework.com/datasets/orinlabs/horizon-1-public)` dataset. To target a single task instead, swap `-d <name>` for `-p evals/<task-dir>`.

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

## How agents and environments are split

Each trial has two pieces in two places:

- **The environment** is a sandbox container (Docker locally, or a Daytona/Modal/etc. sandbox in the cloud). It holds the per-trial filesystem state: `/workdir/trace.jsonl`, the per-tool wrappers `horizon-install-tools` placed in `/usr/local/bin`, the `/state` directory the verifier inspects, etc. The sandbox has no internet access (`allow_internet = false`) and runs no agent code — it only executes shell commands the host sends it.
- **The agent** runs in the `harbor` process on the host (your laptop, a CI runner, wherever you invoked `harbor run`). It owns the conversation history and the `OPENROUTER_API_KEY`, calls the LLM, decides what to do, and dispatches each tool call to its sandbox via `environment.exec(...)`.

When you pass `-n N`, harbor allocates `N` isolated sandboxes AND `N` agent coroutines on the host, all sharing one Python event loop and one HTTP client pool. The sandboxes are perfectly isolated from each other; the agent coroutines are not. **Any synchronous LLM SDK call inside an agent's `run()` will block the event loop and starve the other `N-1` coroutines** — even though their sandboxes are completely independent, they can't dispatch their `environment.exec(...)` calls until the loop is freed. Reference agents in `agents/` use `AsyncOpenAI` and `await` every API call for this reason; if you write your own agent, do the same.