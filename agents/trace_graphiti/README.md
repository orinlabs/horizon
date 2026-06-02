# trace_graphiti

Temporal knowledge-graph memory baseline over `/workdir/trace.jsonl`, backed by [Graphiti](https://github.com/getzep/graphiti) on an embedded [Kuzu](https://kuzudb.com) graph (in-process, no server — the graph analog of `trace_mem0`'s ephemeral Chroma).

Graphiti LLM-extracts entities and relationships from each ingested episode and maintains a **bi-temporal** graph: when a new fact contradicts an older one it *invalidates* the old edge (sets `valid_to`) rather than keeping both. That's the whole point — the recency/override cases (`stale-schedule-vs-actual-pattern`, `mon-thu-5pm-schedule-correction`, `test-rescheduled-to-monday`, `grandfathered-pricing-with-cascading-patches`) punish flat stores that return the stale fact alongside the current one.

## Shape (mirrors `trace_mem0`)

- Ingests the **full** trace, grouped by **UTC day**. Each day → one (or, for long days, a few) episode(s), ingested **chronologically** with `reference_time` set to that day so temporal edge-invalidation fires correctly. Day-grouping (vs mem0's wake/sleep cycles) bounds the episode count — each Graphiti `add_episode` runs extraction + edge-resolution LLM calls, so it's ~an order of magnitude more expensive per episode than a mem0 add.
- Exposes `memory_search` (`graphiti.search` → temporal facts with validity dates) + `shell_exec`. The raw trace is deleted after ingest, so recall must go through the graph.

## Requirements

- `OPENROUTER_API_KEY` + `OPENROUTER_MANAGEMENT_KEY` (per-trial sub-key, like the other agents). LLM, embeddings, and reranking all route through OpenRouter.
- `graphiti-core[kuzu]` in the host env: `uv add 'graphiti-core[kuzu]'`.

## Run

```bash
source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
PYTHONPATH=agents harbor run \
    -p evals/304-13-stale-schedule-vs-actual-pattern-v0 \
    --agent-import-path trace_graphiti.agent:TraceGraphitiAgent \
    -m openai/gpt-4o-mini
```

## Implementation notes

- **Kuzu FTS indices**: `KuzuDriver.setup_schema()` creates the node/edge tables but not the full-text-search indices Graphiti's search needs ([getzep/graphiti#1360](https://github.com/getzep/graphiti/issues/1360)). The agent creates them by hand before ingest (Graphiti's internal entity-resolution searches need them) and rebuilds them after ingest, since Kuzu FTS indices are static snapshots that don't see rows added after creation.
- **Cost accounting**: Graphiti's internal extraction/embedding/rerank calls aren't visible per-response, so the agent reads the per-trial sub-key's full OpenRouter ledger (like the `hermes` agent) for the true total, with a chat-vs-graph_build breakdown in `trajectory.extra`.
- **Fixed graph model**: extraction/dedup/rerank use `GRAPHITI_GRAPH_MODEL` (default `openai/gpt-4o-mini`), independent of the benchmarked chat model, so the built graph is identical across a model sweep.
- **`MAX_EPISODES` cap**: a pathologically long trace is truncated to the most recent days (recency is what the override cases care about) so the sequential ingest stays under the agent timeout.
