# Horizon-1 Public Evaluation Tasks

This directory contains the public example tasks from the Horizon-1 benchmark. The full benchmark is private to prevent overfitting.

## Task Structure

Each task directory follows this structure:

```
<task-name>/
├── task.toml           # Task metadata (name, timeouts, difficulty, etc.)
├── instruction.md      # The instruction given to the agent
├── environment/
│   ├── tools/
│   │   ├── tools.json       # Tool definitions (SDK schema + CLI mapping)
│   │   └── tool_handler.py  # Python handler for tool execution
│   └── workdir/
│       └── trace.jsonl      # Prior-session trace (downloaded at build time)
├── tests/
│   └── judge.py        # Verifier script that checks success criteria
└── solution/           # Oracle solution for validation
```

## Key Files

### task.toml

Defines task metadata:
- `name`: Unique task identifier (e.g., `orinlabs/01-example-catering-vendor`)
- `difficulty`: One of `easy`, `medium`, or `hard`
- `agent.timeout_sec`: Agent execution timeout
- `verifier.timeout_sec`: Verifier execution timeout
- `environment`: Resource constraints (CPUs, memory, storage, network access)

### tools.json

Defines the tools available to the agent in this task:
- `sdk_schema`: OpenAI/OpenRouter function-calling schema
- `handler`: CLI mapping with `argv` and `arg_map` for parameter translation

The tools are installed into `/usr/local/bin` at container startup by `horizon-install-tools`.

### judge.py

A Python script that:
1. Reads the environment state (e.g., `/state/replies.json`)
2. Evaluates success criteria
3. Writes `{"reward": 0|1, "metrics": {...}}` to `/logs/verifier/reward.json`

## Running Tasks

Run a single task:

```bash
harbor run \
  -p evals/01-example-catering-vendor \
  --agent-import-path trace_rag.agent:TraceRagAgent \
  -m openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...
```

Run the full public dataset:

```bash
harbor run \
  -d orinlabs/horizon-1-public \
  -a terminus-2 \
  -m openrouter/openai/gpt-4o-mini \
  --ae OPENROUTER_API_KEY=sk-or-...
```

## Traces

Traces are hosted on HuggingFace at [orinlabs/horizon-1-example-traces](https://huggingface.co/datasets/orinlabs/horizon-1-example-traces) and downloaded into the container at `/workdir/trace.jsonl` during image build.

Each trace is a JSONL file where each line is a JSON object with:
- `timestamp`: ISO-8601 timestamp
- `message_data`: Event payload with `type` field (`message`, `reasoning`, `function_call`, `function_call_output`)

## Dataset Manifest

The `dataset.toml` file registers all public tasks with their content digests for integrity verification:

```toml
[[tasks]]
name = "orinlabs/01-example-catering-vendor"
digest = "sha256:..."
```

This file is used by `harbor publish` to upload tasks to Harbor Hub.
