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

The trace is an append-only JSONL log of timestamped events: messages, reasoning, tool calls, and tool outputs. It is pre-staged at `/workdir/trace.jsonl`.

In `therapy-goals-followthrough`, the trace contains a real tutoring history where Abby's parent sent speech-therapy PDFs. The agent extracted the specific goals, completed the current session, then slept. At eval start, the current environment has a new parent SMS asking for the next session to include those prior goals.

### Environment-Owned Tools

The environment defines the tool surface in `/tools/tools.json`. Agents load this registry and pass the schemas directly to the LLM SDK. Tool calls are routed back through handlers defined by the registry, which mutate environment state under `/state`.

Agents should not invent generic tools for registry-based evals. In particular, API agents should not expose a generic bash tool unless the environment explicitly defines one in `/tools/tools.json`.

The current Acadia eval publishes tools like `sms_list`, `sms_send`, `show_account`, `task_list`, `create_document`, and `update_document`. The private handler backend edits local JSON state; the verifier checks that state after the run.

### Scoring

After `run` returns, Harbor executes the task's `tests/test.sh` inside the container. The script writes either `/logs/verifier/reward.txt` (`1` for pass, `0` for fail) or a richer `/logs/verifier/reward.json`.

All current evals diff final `/state` against the pre-run snapshot and use a strict structured-output LLM judge:

- `therapy-goals-followthrough` scores whether the agent created a concrete, executable plan targeting the specific speech-therapy goals from the trace.
- `mastery-addition-move-on` scores one pivotal action: did the agent send an outbound SMS to the parent that both (a) explicitly names Abby's mastery of addition-with-carrying and (b) names a specific next math topic? State mutations to the stale task/goal/plan doc are captured as sub-metrics but do not gate reward.
- `interests-known-never-used` scores one pivotal artifact: the new session-plan document for today's math lesson. Reward gates on four fields — doc exists with ≥ 3 word problems, ≥ 2 of them use one of Aryan's stated interests as the *subject* of the problem (not just name-dropped in narrative), NO problem uses a generic filler subject (cookies, apples, notebooks, pencils, …), and the problems actually test fractions.

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
HF_TOKEN=hf_...               # read access to orinlabs/horizon-1-eval-traces
```

`HF_TOKEN` is **required to start any eval environment**. Each eval's
`environment/Dockerfile` only contains the static `app/` and `tools/`
scaffolding — `trace.jsonl` is hydrated by a Horizon-specific environment
subclass after the underlying Harbor environment finishes `start()`. The
subclass downloads `<eval-slug>/trace.jsonl` from the private HF dataset
[`orinlabs/horizon-1-eval-traces`](https://huggingface.co/datasets/orinlabs/horizon-1-eval-traces)
on the host and uploads it into the container at `/workdir/trace.jsonl`.
By the time any agent is dropped in, the environment is fully ready.

The repo ships a `harbor.yaml` that wires this in by default (using the
local Docker driver). Pass `--config harbor.yaml` to every `harbor run`:

```bash
PYTHONPATH=agents harbor run --config harbor.yaml \
    -p evals/01-direct-semantic-holiday-party-caterer \
    --agent-import-path trace_rag.agent:TraceRagAgent \
    -m google/gemini-2.5-flash -y
```

To run on Modal instead, override the env subclass on the CLI:

```bash
PYTHONPATH=agents harbor run --config harbor.yaml \
    --environment-import-path horizon_environment:HorizonModalEnvironment \
    -p evals/01-direct-semantic-holiday-party-caterer \
    --agent-import-path trace_rag.agent:TraceRagAgent \
    -m google/gemini-2.5-flash -y
```

> Note: `scripts/fetch_eval_data.py` is also available as a host-side fallback
> for inspecting traces without docker, or for fetching `*.raw.jsonl` source
> files (`--raw`) when you need to re-run an eval's `scripts/build_trace.py`.

Run the registry-native trace dump baseline:

```bash
source .env && export OPENROUTER_API_KEY HF_TOKEN
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-direct-semantic-holiday-party-caterer \
    --agent-import-path trace_dump.agent:TraceDumpAgent \
    -m google/gemini-2.5-flash \
    -y
```

Run oracle and nop checks (these don't need the trace, so any `--env` is fine):

```bash
source .env && export OPENROUTER_API_KEY HF_TOKEN
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-direct-semantic-holiday-party-caterer -a oracle -y
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-direct-semantic-holiday-party-caterer -a nop -y
```

Every run writes per-trial trajectories, verifier logs, and artifacts under `jobs/<timestamp>/`.

## Included Agents

- [`trace_dump`](agents/trace_dump/) — dumps the whole trace into context, then uses environment-owned tools from `/tools/tools.json`.
- [`trace_rag`](agents/trace_rag/) — chunks the trace by day, exposes an agent-owned `trace_search` tool, and combines it with environment-owned tools.
- [`hermes`](agents/hermes/) — installed-agent wrapper that seeds Hermes's native session DB from the trace.

## Writing Registry-Based Agents

A registry-based API agent should:

1. Read or index `/workdir/trace.jsonl`.
2. Load `/tools/tools.json`.
3. Pass the registry's `sdk_schema` entries directly to the LLM SDK.
4. Route model tool calls through the registry handler definitions.
5. Emit ATIF trajectory data if possible.

See [`agents/environment_tools.py`](agents/environment_tools.py) and [`agents/trace_dump/agent.py`](agents/trace_dump/agent.py) for the minimal implementation.

## Inspect Results

Harbor ships a browser UI for trial trajectories and verifier output:

```bash
harbor view jobs
```
