"""Shared helpers used by every reference agent.

Kept intentionally tiny — only things that are duplicated *and* whose
behavior we want to evolve in lockstep across agents (so a fix in one
place fixes them all). Anything agent-specific (system prompts, event
formatters, retrieval indexes, exec retry wrappers tied to per-agent
truncation) stays inline in that agent.
"""

from __future__ import annotations

import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any

from harbor.environments.base import BaseEnvironment


def usage_cost(resp: Any) -> float:
    """Extract per-call USD cost from an OpenRouter chat/embedding response.

    Requires the request to have included ``extra_body={"usage": {"include": True}}``
    so OpenRouter attaches ``usage.cost`` (in USD) to the response. Returns
    0.0 when the provider didn't surface a cost (non-OpenRouter providers,
    older models, transient billing pipeline gaps).

    Resilient to where the OpenAI Python SDK parks unknown fields:
    direct attr, ``model_extra``, or ``model_dump()``.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0.0
    cost = getattr(usage, "cost", None)
    if cost is None:
        extra = getattr(usage, "model_extra", None) or {}
        cost = extra.get("cost")
    if cost is None and hasattr(usage, "model_dump"):
        cost = usage.model_dump().get("cost")
    try:
        return float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


@asynccontextmanager
async def timed_call(call_log: list[dict] | None, kind: str, label: str):
    """Record wall-clock for an awaitable hot-spot into ``call_log``.

    Use to break down where seconds go inside an agent run. ``kind`` is a
    coarse bucket (``"exec"``, ``"embedding"``, ``"chat"``, ``"mem0"``,
    etc.) so we can aggregate per kind. ``label`` is free-form for
    drill-down (``"cat trace.jsonl"``, ``"chat turn 4"``).

    Records ``elapsed_s`` even if the wrapped block raises. The exception
    is re-raised after recording, with the exception class name stored in
    ``error``. Pass ``call_log=None`` to disable recording cheaply.

    Usage::

        call_log: list[dict] = []
        async with timed_call(call_log, "exec", "cat trace"):
            payload = await env.exec("cat /workdir/trace.jsonl")
    """
    if call_log is None:
        yield
        return
    t0 = time.monotonic()
    err: str | None = None
    try:
        yield
    except BaseException as exc:
        err = type(exc).__name__
        raise
    finally:
        call_log.append(
            {
                "kind": kind,
                "label": label,
                "elapsed_s": round(time.monotonic() - t0, 3),
                "error": err,
            }
        )


async def read_trace_file(
    environment: BaseEnvironment,
    remote_path: str,
) -> str:
    """Read a (potentially large) file from the sandbox into a host string.

    Uses ``environment.download_file()`` rather than piping through
    ``exec`` + stdout capture. This is critical for the prior-session
    trace files (``/workdir/trace.jsonl``), which can exceed 20MB on
    real long-horizon tasks. Routing those through ``exec`` truncates
    them at whatever the agent's stdout cap is, silently losing the
    bulk of the prior session and giving retrieval/summarization layers
    only the most recent few KB to work with.

    Returns the file's full UTF-8 contents, or ``""`` if the file does
    not exist in the sandbox (matching the historical ``cat ... 2>/dev/null
    || true`` behavior so callers don't need to re-handle "missing trace").
    """
    fd, local_path = tempfile.mkstemp(prefix="horizon_trace_", suffix=".jsonl")
    os.close(fd)
    try:
        try:
            await environment.download_file(remote_path, local_path)
        except Exception:
            return ""
        try:
            with open(local_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except FileNotFoundError:
            return ""
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass


def summarize_call_log(call_log: list[dict]) -> dict[str, Any]:
    """Aggregate a call log into per-kind {count, sum_s, mean_s, max_s}.

    Suitable to drop directly into ``trajectory.extra``. The full
    ``call_log`` is also worth keeping alongside for drill-down.
    """
    by_kind: dict[str, list[float]] = {}
    for entry in call_log:
        by_kind.setdefault(entry["kind"], []).append(float(entry["elapsed_s"]))
    out: dict[str, Any] = {}
    for kind, secs in by_kind.items():
        n = len(secs)
        out[kind] = {
            "count": n,
            "sum_s": round(sum(secs), 3),
            "mean_s": round(sum(secs) / n, 3),
            "max_s": round(max(secs), 3),
        }
    return out
