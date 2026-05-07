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

## What a trace looks like

Each trace is a [JSONL](https://jsonlines.org/) file at `/workdir/trace.jsonl`. One JSON object per line, each with a `timestamp` and a `message_data` payload. The payload mirrors the OpenAI Responses item shape — a `message` (system/user/assistant turn), a `reasoning` summary, a `function_call`, or a `function_call_output` — so any agent that already speaks that schema can ingest a trace with no translation layer.

A short excerpt from `01-example-catering-vendor`:

```jsonl
{"timestamp": "2026-04-08T14:12:00.000000+00:00", "message_data": {"role": "system", "type": "message", "content": "You are being woken up from sleep. Gather as much information as possible about what's happening. …"}}
{"timestamp": "2026-04-08T14:12:13.044556+00:00", "message_data": {"id": null, "type": "reasoning", "summary": "Just woke up. I should pull the inbox and the task list in parallel so I have a complete picture of what's open before I start acting on anything specific."}}
{"timestamp": "2026-04-08T14:12:14.217164+00:00", "message_data": {"id": "toolu_qv12RvzCjTPAlLbIzP8JXk", "name": "email_list", "type": "function_call", "call_id": "toolu_qv12RvzCjTPAlLbIzP8JXk", "arguments": "{\"reasoning\": \"Inbox sweep on wake.\", \"limit\": 10}"}}
{"timestamp": "2026-04-08T14:12:14.221833+00:00", "message_data": {"type": "function_call_output", "call_id": "toolu_qv12RvzCjTPAlLbIzP8JXk", "output": "INBOX (3 threads)\n- bookings@cedarpointlodge.example  thread_id: th-offsite-venue  …\n- Rosa Lin <rosa@northpierpantry.example>  thread_id: th-northpier  …\n- Jordan Lee <jordan@company.example>  thread_id: th-jordan-budget  …"}}
{"timestamp": "2026-04-08T14:12:15.032220+00:00", "message_data": {"id": "toolu_euJjskTCSbhYR7Fwb3wwm2", "name": "email_read", "type": "function_call", "call_id": "toolu_euJjskTCSbhYR7Fwb3wwm2", "arguments": "{\"reasoning\": \"Jordan's note is most time-sensitive.\", \"thread_id\": \"th-jordan-budget\"}"}}
{"timestamp": "2026-04-08T14:12:15.033942+00:00", "message_data": {"type": "function_call_output", "call_id": "toolu_euJjskTCSbhYR7Fwb3wwm2", "output": "THREAD th-jordan-budget\nSubject: Re: June team dinner — please pick a vendor by Friday\nFrom: Jordan Lee <jordan@company.example>\nCasey — please pick the caterer for the June team dinner by end of Friday so finance can cut the deposit. Budget ceiling is $1,400 all-in."}}
```

Production traces can be tens of MB and tens of thousands of events; the public examples are smaller. Browse the full set at [orinlabs/horizon-1-example-traces](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces).


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