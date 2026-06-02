# trace_rlm

Recursive Language Model (RLM) baseline over `/workdir/trace.jsonl`, following [Zhang, Kraska & Khattab (2026)](https://arxiv.org/abs/2512.24601) ([blog](https://alexzhang13.github.io/blog/2025/rlm/)).

Instead of stuffing the (2–36M token) trace into the model's context window, the trace is loaded as a **variable in a persistent Python REPL**. A *root* LM sees only the task + the trace's size, and writes code to peek, grep, partition, and launch **recursive depth-1 sub-LM calls** over slices of the trace. No single LM call holds the whole trace, which sidesteps "context rot" and scales past the context window.

It's the recursive cousin of `trace_shell_context` (greps the raw trace but no recursion, whole file in context) and `trace_window` (recent slice only): RLM keeps the trace out of the root context entirely and lets the model decide how to decompose it at test time.

## Tools (root LM)

- `repl_exec(code)` — run Python in a persistent notebook pre-loaded with:
  - `trace: str` — full transcript
  - `lines: list[str]` — raw JSONL event lines
  - `events: list[dict]` — parsed events
  - `llm(prompt, system=None) -> str` — one cheap sub-LM call
  - `recurse(context, query) -> str` — ask the sub-LM `query` about a `context` slice (the main recursion primitive)
- `shell_exec(command)` — act on the task environment.

## Root vs recursive model

Per the paper, the root and recursive LMs differ: a capable root model (`-m ...`) drives the REPL while a cheaper `RLM_SUB_MODEL` (default `openai/gpt-5-mini`) answers recursive sub-queries. Recursion depth is fixed at 1 (root calls LMs, not other RLMs). All calls route through the per-trial OpenRouter sub-key; cost is split `root` vs `recursive` in `trajectory.extra`.

## Requirements

- `OPENROUTER_API_KEY` + `OPENROUTER_MANAGEMENT_KEY`. No extra packages — pure OpenRouter chat completions.

## Run

```bash
source .env && export OPENROUTER_API_KEY OPENROUTER_MANAGEMENT_KEY
PYTHONPATH=agents harbor run \
    -p evals/214-30-original-scope-recall-v0 \
    --agent-import-path trace_rlm.agent:TraceRlmAgent \
    -m openai/gpt-5
```

## Implementation notes

- **REPL safety/timeout**: model-generated code runs via `exec()` in a daemon worker thread with a per-cell wall-clock timeout. A timed-out cell can't be force-killed (it finishes in the background); the root LM is told to write a cheaper cell. The trace lives in the host REPL, and `/workdir/trace.jsonl` is deleted from the sandbox so `shell_exec` can't bypass the RLM by cat-ing it.
- **Sub-LM calls**: `llm`/`recurse` use a synchronous OpenAI client (the cell runs off the event loop), accumulating token/cost usage under a lock since one cell can fan out many calls.
- **Higher `MAX_STEPS`**: RLM trajectories (peek → grep → map → answer → act) are longer than a flat retrieval loop.
