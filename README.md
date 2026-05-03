# Horizon-1

![Horizon-1](docs/benchmark.png)

[![Stars](https://img.shields.io/github/stars/orinlabs/horizon-1?style=flat&logo=github)](https://github.com/orinlabs/horizon-1/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/orinlabs/horizon-1)](https://github.com/orinlabs/horizon-1/commits/main)
[![License](https://img.shields.io/github/license/orinlabs/horizon-1)](./LICENSE)
[![Harbor](https://img.shields.io/badge/harness-harbor-blue)](https://www.harborframework.com/)

A continual learning benchmark for extremely long-horizon agents, packaged as [Harbor](https://www.harborframework.com/) tasks and agents.

## Purpose

As agents get more autonomous, their ability to learn continuously has become a critical bottleneck for usefulness. Existing memory benchmarks ([LoCoMo](https://arxiv.org/abs/2402.17753), [LongMemEval](https://arxiv.org/abs/2410.10813)) measure reactive chatbot applications, not autonomous agents. Existing learning benchmarks like [ARC-AGI](https://arcprize.org/) measure acquisition and application, but use sandboxed environments that are not representative of knowledge work.

Horizon-1 measures whether an agent can acquire learnings from a long first-person history and apply them later in a stateful environment. It makes no distinction between models and harnesses: the target is the utility of the learning system, regardless of how it is crafted.

## Anatomy of an Eval

Every Horizon-1 eval is a `(trace, environment)` pair. The agent ingests a long first-person history (the **trace**), then takes live actions in the **environment** to finish whatever work is still pending. A verifier scores the final state.

### Trace

The trace is an append-only JSONL log of timestamped events: messages, reasoning, tool calls, and tool outputs. Example traces are synthetic and hosted in the public Hugging Face dataset [`orinlabs/horizon-1-example-traces`](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces).

At eval start, the Horizon environment subclass downloads `<eval-slug>/trace.jsonl` from Hugging Face and stages it at `/workdir/trace.jsonl`. Agents can read or index that file before it is removed by retrieval-style baselines.

### Environment-Owned Tools

The environment defines the tool surface in `/tools/tools.json`. Agents load this registry and pass the schemas directly to the LLM SDK. Tool calls are routed back through handlers defined by the registry, which mutate environment state under `/state`.

Agents should not invent generic tools for registry-based evals. In particular, API agents should not expose a generic bash tool unless the environment explicitly defines one in `/tools/tools.json`.

The example evals publish a small inbox API: `inbox_list`, `inbox_read`, and `reply_send`. The handler backend edits local JSON state; the verifier checks that state after the run.

### Scoring

After `run` returns, Harbor executes the task's `tests/test.sh` inside the container. The script writes either `/logs/verifier/reward.txt` (`1` for pass, `0` for fail) or a richer `/logs/verifier/reward.json`.

The example evals use deterministic Python judges. Each one checks that the agent replied to the right thread with details that are only available in the prior trace, while avoiding distractor details from the current inbox.

## Getting Started

Install Harbor:

```bash
git clone https://github.com/orinlabs/horizon-1.git
cd horizon-1
uv tool install harbor
```

Configure secrets:

```bash
cp .env.example .env
```

Then edit `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

Example traces are public, so no Hugging Face token is needed. Run an example
with the Horizon Docker environment, which hydrates `/workdir/trace.jsonl`
before the agent starts:

```bash
source .env && export OPENROUTER_API_KEY
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-example-catering-vendor \
    --agent-import-path trace_keyword.agent:TraceKeywordAgent \
    -m openai/gpt-4o-mini \
    -y
```

To run on Modal instead, use the Modal environment subclass:

```bash
source .env && export OPENROUTER_API_KEY
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonModalEnvironment \
    -p evals/01-example-catering-vendor \
    --agent-import-path trace_keyword.agent:TraceKeywordAgent \
    -m openai/gpt-4o-mini \
    -y
```

You can also fetch traces locally for inspection:

```bash
uv run --with huggingface_hub python scripts/fetch_eval_data.py
```

Run oracle and nop checks:

```bash
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-example-catering-vendor -a oracle -y
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-example-catering-vendor -a nop -y
```

Included examples:

- `01-example-catering-vendor`: recover the selected event vendor and a supporting quote detail.
- `02-example-expense-code`: recover a travel reimbursement project code and receipt requirement.
- `03-example-reading-preference`: recover a reading-group paper preference and note format.

Every run writes per-trial trajectories, verifier logs, and artifacts under `jobs/<timestamp>/`.

## Included Agents

- [`trace_window`](agents/trace_window/) - reads a bounded trace window, then uses environment-owned tools from `/tools/tools.json`.
- [`trace_keyword`](agents/trace_keyword/) - chunks the trace by day, exposes a BM25 `trace_search` tool, and combines it with environment-owned tools.
- [`trace_rag`](agents/trace_rag/) - chunks the trace by day, exposes an embedding-backed `trace_search` tool, and combines it with environment-owned tools.

## Writing Registry-Based Agents

A registry-based API agent should:

1. Read or index `/workdir/trace.jsonl`.
2. Load `/tools/tools.json`.
3. Pass the registry's `sdk_schema` entries directly to the LLM SDK.
4. Route model tool calls through the registry handler definitions.
5. Emit ATIF trajectory data if possible.

See [`agents/environment_tools.py`](agents/environment_tools.py) and [`agents/trace_keyword/agent.py`](agents/trace_keyword/agent.py) for a minimal retrieval implementation.

## Inspect Results

Harbor ships a browser UI for trial trajectories and verifier output:

```bash
harbor view jobs
```
