# trace_rag

RAG baseline over `/workdir/trace.jsonl`. Chunks the trace by UTC day, embeds each day-block with OpenAI's `text-embedding-3-small`, and exposes a `trace_search(query, k)` tool the LLM can call instead of dumping the whole trace into context.

The task environment must publish `/tools/tools.json`; `trace_rag` combines the agent-owned `trace_search` tool with those environment-owned tools. It does not provide a generic bash tool.

## How it differs from prompt stuffing

`trace_rag` does not put the full trace in the system prompt. It embeds day chunks once at ingest, exposes `trace_search(query, k=3)`, and keeps later chat turns focused on retrieved chunks rather than the entire trace.

## Requirements

- `OPENROUTER_API_KEY` — same as every other agent in this repo. OpenRouter proxies both chat completions and embeddings (model slug `openai/text-embedding-3-small`).

## Run

```bash
source .env && export OPENROUTER_API_KEY
PYTHONPATH=agents harbor run \
    --environment-import-path horizon_environment:HorizonDockerEnvironment \
    -p evals/01-example-catering-vendor \
    --agent-import-path trace_rag.agent:TraceRagAgent \
    -m openai/gpt-4o-mini
```

## Implementation notes

- **Per-day chunking**: events are grouped by UTC date extracted from the event's `timestamp`. Real long-horizon traces can span weeks or months, producing many day chunks.
- **Trace file deleted after ingest**: once the chunks are embedded, the agent removes `/workdir/trace.jsonl` from the sandbox. The LLM can only recall prior history through `trace_search`.
- **In-memory store**: chunks and L2-normalized embeddings live in a numpy array on the agent instance. No persistence between trials; each trial re-embeds.
- **Cosine similarity**: `matrix @ query` where both are unit-normalized.
- **Embedding tokens tracked**: accumulated into the ATIF `trajectory.extra.embedding_tokens` field. Chat tokens go into `final_metrics` as usual.
