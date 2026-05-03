# Horizon-1

![Horizon-1](docs/benchmark.png)

[![Stars](https://img.shields.io/github/stars/orinlabs/horizon-1?style=flat&logo=github&cacheSeconds=300)](https://github.com/orinlabs/horizon-1/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/orinlabs/horizon-1?cacheSeconds=300)](https://github.com/orinlabs/horizon-1/commits/main)
[![License](https://img.shields.io/github/license/orinlabs/horizon-1?cacheSeconds=300)](./LICENSE)
[![Harbor](https://img.shields.io/badge/harness-harbor-blue)](https://www.harborframework.com/)

A continual learning benchmark for extremely long-horizon agents, packaged as [Harbor](https://www.harborframework.com/) tasks and agents.

## Purpose

As agents get more autonomous, their ability to learn continuously has become a critical bottleneck for usefulness. Existing memory benchmarks ([LoCoMo](https://arxiv.org/abs/2402.17753), [LongMemEval](https://arxiv.org/abs/2410.10813)) measure reactive chatbot applications, not autonomous agents. Existing learning benchmarks like [ARC-AGI](https://arcprize.org/) measure acquisition and application, but use sandboxed environments that are not representative of knowledge work.

Horizon-1 measures whether an agent can acquire learnings from a long first-person history and apply them later in a stateful environment. It makes no distinction between models and harnesses: the target is the utility of the learning system, regardless of how it is crafted.

Per-task prior-session traces (the long first-person history each task starts with) live in the public Hugging Face dataset [`orinlabs/horizon-1-example-traces`](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces) and are pulled into each eval's environment image at build time.

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

# reference agent shipped in this repo
harbor run \
  -d orinlabs/horizon-1-public \
  --agent-import-path trace_shell_context.agent:TraceShellContextAgent \
  -m openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...
```

`-d` runs every task in the [`orinlabs/horizon-1-public`](https://hub.harborframework.com/datasets/orinlabs/horizon-1-public) dataset. To target a single task instead, swap `-d <name>` for `-p evals/<task-dir>`.

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

