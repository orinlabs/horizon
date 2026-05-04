# trace_rag

RAG baseline over `/workdir/trace.jsonl`. Chunks the trace by UTC day, embeds each day-block with OpenAI's `text-embedding-3-small`, and exposes a `trace_search(query, k)` tool the LLM can call instead of dumping the whole trace into context.

For acting on the current world the agent exposes a single `shell_exec` tool, mirroring `trace_shell_context`. The eval's Dockerfile installs per-tool wrappers in `/usr/local/bin` (e.g. `inbox_list`, `reply_send`) via `horizon-install-tools`, so the model invokes them as plain shell commands — same names and flags it can see in retrieved trace chunks.

## How it differs from `trace_shell_context`

| | `trace_shell_context` | `trace_rag` |
|---|---|---|
| Trace in system prompt? | Yes (entire file) | No |
| Extra tool | — | `trace_search(query, k=3)` |
| API calls for ingest | 0 | 1 batch embedding of N day-chunks |
| Token cost scales with | Trace size × every LLM turn | Query size × search calls |

On a 2-message trace `trace_shell_context` is cheaper. On a multi-week real trace, RAG wins by orders of magnitude.

## Requirements

- `OPENROUTER_API_KEY` — same as every other agent in this repo. OpenRouter proxies both chat completions and embeddings (model slug `openai/text-embedding-3-small`).

## Run

```bash
source .env && export OPENROUTER_API_KEY
PYTHONPATH=agents harbor run \
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
