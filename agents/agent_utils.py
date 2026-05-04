"""Shared helpers used by every reference agent.

Kept intentionally tiny — only things that are duplicated *and* whose
behavior we want to evolve in lockstep across agents (so a fix in one
place fixes them all). Anything agent-specific (system prompts, event
formatters, retrieval indexes, exec retry wrappers tied to per-agent
truncation) stays inline in that agent.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
import time
from collections.abc import Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from harbor.environments.base import BaseEnvironment


DEFAULT_TOOL_REGISTRY_PATH = "/.horizon/tools/tools.json"
DEFAULT_TOOL_OUTPUT_CHAR_CAP = 12_000


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


@dataclass(frozen=True)
class HorizonTool:
    """A single tool installed in the sandbox by ``horizon-install-tools``.

    Mirrors one entry of ``/.horizon/tools/tools.json``:

    - ``sdk_schema`` is the OpenAI/OpenRouter ``{"type": "function", ...}``
      tool descriptor — pass straight through to ``tools=[...]`` on chat
      completions.
    - ``argv`` + ``arg_map`` describe the equivalent CLI invocation. Each
      tool has a wrapper at ``/usr/local/bin/<name>`` that just shells
      through to the registry's ``tool_handler.py``, so calling the tool
      is literally running ``<name> --flag value ...`` in the sandbox.
    """

    name: str
    sdk_schema: dict[str, Any]
    argv: list[str]
    arg_map: dict[str, str]
    handler_type: str = "command"


@dataclass
class HorizonToolRegistry:
    """All tools loaded from the sandbox's ``tools.json``.

    Exposes the OpenRouter-compatible ``tools`` list for chat completions
    and a single ``call(...)`` entry point that dispatches a model's
    function call back to the matching CLI command in the sandbox.
    """

    tools: list[HorizonTool]
    schema_version: str = "horizon-tools-v1"
    _by_name: dict[str, HorizonTool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_name = {t.name: t for t in self.tools}

    @property
    def openrouter_tools(self) -> list[dict[str, Any]]:
        """The list to pass as ``tools=`` to ``client.chat.completions.create``."""
        return [t.sdk_schema for t in self.tools]

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get(self, name: str) -> HorizonTool | None:
        return self._by_name.get(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._by_name

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self) -> Iterable[HorizonTool]:
        return iter(self.tools)

    async def call(
        self,
        environment: BaseEnvironment,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        timeout_sec: int = 60,
        output_char_cap: int = DEFAULT_TOOL_OUTPUT_CHAR_CAP,
        retries: int = 3,
    ) -> dict[str, Any]:
        """Dispatch a function call to the matching sandbox CLI command.

        Returns a dict shaped like the agents' existing ``shell_exec``
        output: ``{"exit_code", "stdout", "stderr", "command"}``. Unknown
        tool names return ``exit_code=127`` with the error in ``stderr``,
        matching the established ``shell_exec`` fallback so callers don't
        need to special-case it before stuffing the payload back into the
        chat history.
        """
        tool = self._by_name.get(name)
        if tool is None:
            return {
                "exit_code": 127,
                "stdout": "",
                "stderr": f"unknown tool: {name}",
                "command": "",
            }
        return await call_tool(
            environment,
            tool,
            arguments or {},
            timeout_sec=timeout_sec,
            output_char_cap=output_char_cap,
            retries=retries,
        )


def _parse_tool_entry(entry: Mapping[str, Any]) -> HorizonTool:
    """Parse one ``tools.json`` entry into a ``HorizonTool``.

    Raises ``ValueError`` with a precise message on schema problems so a
    broken environment image fails loudly at agent startup instead of
    silently dropping tools the model is supposed to be able to call.
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"tools.json entry missing 'name': {entry!r}")
    sdk_schema = entry.get("sdk_schema")
    if not isinstance(sdk_schema, dict):
        raise ValueError(f"tools.json entry {name!r} missing 'sdk_schema'")
    handler = entry.get("handler") or {}
    if not isinstance(handler, dict):
        raise ValueError(f"tools.json entry {name!r} has non-object 'handler'")
    handler_type = str(handler.get("type") or "command")
    if handler_type != "command":
        raise ValueError(
            f"tools.json entry {name!r} has unsupported handler type "
            f"{handler_type!r} (only 'command' is supported)"
        )
    argv = handler.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError(
            f"tools.json entry {name!r} has invalid handler.argv: {argv!r}"
        )
    arg_map_raw = handler.get("arg_map") or {}
    if not isinstance(arg_map_raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in arg_map_raw.items()
    ):
        raise ValueError(
            f"tools.json entry {name!r} has invalid handler.arg_map: {arg_map_raw!r}"
        )
    return HorizonTool(
        name=name,
        sdk_schema=sdk_schema,
        argv=list(argv),
        arg_map=dict(arg_map_raw),
        handler_type=handler_type,
    )


def parse_tool_registry(payload: Mapping[str, Any]) -> HorizonToolRegistry:
    """Parse a decoded ``tools.json`` document into a ``HorizonToolRegistry``."""
    raw_tools = payload.get("tools")
    if not isinstance(raw_tools, list):
        raise ValueError("tools.json: top-level 'tools' must be a list")
    parsed = [_parse_tool_entry(t) for t in raw_tools]
    schema_version = str(payload.get("schema_version") or "horizon-tools-v1")
    return HorizonToolRegistry(tools=parsed, schema_version=schema_version)


async def load_environment_tools(
    environment: BaseEnvironment,
    *,
    registry_path: str = DEFAULT_TOOL_REGISTRY_PATH,
) -> HorizonToolRegistry:
    """Download ``tools.json`` from the sandbox and parse it into a registry.

    Uses ``environment.download_file`` rather than ``exec``+``cat`` so
    we don't truncate the registry at the agent's stdout cap and so we
    surface a real error if the file is missing instead of returning an
    empty string. Most evals install a handful of small tools; large
    registries are still well under any reasonable file-transfer limit.
    """
    fd, local_path = tempfile.mkstemp(prefix="horizon_tools_", suffix=".json")
    os.close(fd)
    try:
        await environment.download_file(registry_path, local_path)
        with open(local_path, encoding="utf-8") as fh:
            payload = json.load(fh)
    finally:
        try:
            os.unlink(local_path)
        except OSError:
            pass
    if not isinstance(payload, dict):
        raise ValueError(
            f"{registry_path}: expected a JSON object, got {type(payload).__name__}"
        )
    return parse_tool_registry(payload)


def _property_type(tool: HorizonTool, key: str) -> str | None:
    """Look up the JSON-schema ``type`` for ``key`` in the tool's parameters.

    Used by :func:`build_tool_command` to choose between argparse's two
    common idioms for booleans and arrays: ``store_true`` (just emit the
    flag) vs. ``--flag <value>`` (emit flag + stringified value).
    """
    fn = tool.sdk_schema.get("function") if isinstance(tool.sdk_schema, dict) else None
    if not isinstance(fn, dict):
        return None
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if not isinstance(params, dict):
        return None
    props = params.get("properties")
    if not isinstance(props, dict):
        return None
    prop = props.get(key)
    if not isinstance(prop, dict):
        return None
    t = prop.get("type")
    return t if isinstance(t, str) else None


def build_tool_command(
    tool: HorizonTool,
    arguments: Mapping[str, Any] | None = None,
) -> str:
    """Render a function-call's arguments into the equivalent shell command.

    The command is exactly what the model would type if it were driving
    the sandbox by hand: ``<tool> --flag value --flag value`` etc. Every
    fragment is run through ``shlex.quote`` so freeform string args
    (subjects, message bodies, code snippets) survive the shell intact.

    Conventions:

    - ``None`` values are dropped (use the handler's default).
    - Booleans use argparse's ``store_true`` idiom: emit the flag iff
      ``True``. Override by giving the schema ``type: "string"`` and
      passing ``"true"``/``"false"`` if you really want a value form.
    - Arrays / objects are JSON-encoded and passed as a single value.
    - Args that aren't in ``arg_map`` get a best-effort ``--kebab-case``
      flag — this lets the model pass forward-compatible extras the
      registry author hasn't enumerated yet.
    """
    parts: list[str] = list(tool.argv)
    for key, value in (arguments or {}).items():
        if value is None:
            continue
        flag = tool.arg_map.get(key) or f"--{key.replace('_', '-')}"
        prop_type = _property_type(tool, key)
        if prop_type == "boolean" or isinstance(value, bool):
            if bool(value):
                parts.append(flag)
            continue
        if isinstance(value, (list, dict)) or prop_type in {"array", "object"}:
            parts.extend([flag, json.dumps(value, separators=(",", ":"))])
            continue
        parts.extend([flag, str(value)])
    return " ".join(shlex.quote(p) for p in parts)


async def call_tool(
    environment: BaseEnvironment,
    tool: HorizonTool,
    arguments: Mapping[str, Any] | None = None,
    *,
    timeout_sec: int = 60,
    output_char_cap: int = DEFAULT_TOOL_OUTPUT_CHAR_CAP,
    retries: int = 3,
) -> dict[str, Any]:
    """Execute ``tool`` in the sandbox with ``arguments`` and capture output.

    Returns a payload shaped like the existing ``shell_exec`` results
    (``exit_code``/``stdout``/``stderr``) plus the rendered ``command``
    string, so trajectory recorders can store the literal CLI invocation
    next to the LLM-side function call.

    Retries transient ``environment.exec`` failures with exponential
    backoff (1s, 2s, 4s, ...). The terminal failure is reported as a
    non-zero exit with the error chain in ``stderr`` rather than raising
    — matching how the reference agents already feed exec failures back
    to the model.
    """
    command = build_tool_command(tool, arguments)
    last_error = ""
    for attempt in range(max(1, retries) + 1):
        if attempt:
            await asyncio.sleep(2 ** (attempt - 1))
        try:
            result = await environment.exec(command, timeout_sec=timeout_sec)
            return {
                "exit_code": result.return_code,
                "stdout": (result.stdout or "")[-output_char_cap:],
                "stderr": (result.stderr or "")[-output_char_cap:],
                "command": command,
            }
        except Exception as exc:
            last_error = f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
    return {
        "exit_code": 1,
        "stdout": "",
        "stderr": last_error,
        "command": command,
    }


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


# ---------------------------------------------------------------------------
# OpenRouter sub-key helpers — per-trial cost isolation.
#
# When OPENROUTER_PROVISIONING_KEY is set, agents mint a disposable child
# key per trial, route every LLM call through it, snapshot usage on exit,
# and delete the child key. trajectory.extra.cost_usd.total then reflects
# *exact* USD billed for the trial — including subprocess CLIs and
# sub-libraries that have their own OpenAI clients (mem0, lcm-tui, hermes
# chat, openclaw agent).
#
# When the provisioning key isn't set, agents fall back to the shared
# OPENROUTER_API_KEY. Per-trial cost can't then be isolated from
# concurrent traffic, and ``cost_usd.mode`` is set to ``"shared_key"``
# to document the limitation.
# ---------------------------------------------------------------------------

OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/keys"


@dataclass
class TrialKeyState:
    """Mutable state passed through :func:`trial_subkey`.

    Read ``usage_usd`` and ``mode`` AFTER the ``async with`` block exits.
    """

    key: str  # use this for LLM calls (host or sandbox)
    mode: str  # "isolated_subkey" | "shared_key" | "isolated_subkey_query_failed"
    hash: str | None = None
    usage_usd: float | None = None


async def provision_subkey(
    provisioning_key: str, *, label: str, limit_usd: float | None = 5.00
) -> dict[str, str]:
    """Mint a disposable OpenRouter API key. Returns ``{"key": str, "hash": str}``.

    ``label`` is a human-readable name (visible in the OpenRouter dashboard).
    ``limit_usd`` caps total spend on this child key — runaway protection.
    Caller MUST call :func:`delete_subkey` when done.
    """
    import httpx

    body: dict[str, Any] = {"name": label}
    if limit_usd is not None:
        body["limit"] = limit_usd
    headers = {
        "Authorization": f"Bearer {provisioning_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(OPENROUTER_KEYS_URL, headers=headers, json=body)
        resp.raise_for_status()
        payload = resp.json()
    data = payload.get("data") or {}
    return {"key": payload["key"], "hash": data["hash"]}


async def subkey_usage(provisioning_key: str, key_hash: str) -> float:
    """Return cumulative USD spent on a child key (queried via its hash)."""
    import httpx

    headers = {"Authorization": f"Bearer {provisioning_key}"}
    url = f"{OPENROUTER_KEYS_URL}/{key_hash}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()
    data = payload.get("data") or {}
    return float(data.get("usage", 0.0))


async def delete_subkey(provisioning_key: str, key_hash: str) -> bool:
    """Best-effort delete. Returns True on 2xx, False otherwise. Idempotent."""
    import httpx

    headers = {"Authorization": f"Bearer {provisioning_key}"}
    url = f"{OPENROUTER_KEYS_URL}/{key_hash}"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.delete(url, headers=headers)
            return 200 <= resp.status_code < 300
    except Exception:
        return False


def cost_note(mode: str) -> str:
    """Stable user-facing note for ``trajectory.extra.cost_usd._note``."""
    if mode == "isolated_subkey":
        return (
            "Total = cumulative USD on a per-trial OpenRouter sub-key. "
            "Captures every call billed to the sub-key (agent direct calls, "
            "sub-library internals, subprocess CLIs). Sub-key is created at "
            "trial start and deleted at trial end."
        )
    if mode == "isolated_subkey_query_failed":
        return (
            "Sub-key was minted but post-trial usage query failed; sub-key "
            "deleted (best-effort). No cost data available."
        )
    if mode == "shared_key":
        return (
            "OPENROUTER_PROVISIONING_KEY not set; trial used the shared "
            "OPENROUTER_API_KEY. Per-trial cost cannot be isolated from "
            "concurrent traffic. Set OPENROUTER_PROVISIONING_KEY for exact "
            "per-trial USD."
        )
    return ""


@asynccontextmanager
async def trial_subkey(
    *,
    provisioning_key: str | None,
    fallback_key: str,
    label: str,
    limit_usd: float | None = 5.00,
    settle_seconds: float = 5.0,
):
    """Yield a :class:`TrialKeyState`. Mutates ``usage_usd`` + ``mode`` after exit.

    If ``provisioning_key`` is set: mints a sub-key for the trial, hands it
    to the caller as ``state.key``, snapshots usage on exit, deletes the
    sub-key. ``state.usage_usd`` will be the exact USD spent through that
    sub-key (covering everything billed to it — agent direct calls,
    sub-libraries, subprocess CLIs).

    If ``provisioning_key`` is None: falls back to ``fallback_key``.
    ``usage_usd`` stays None (no per-trial attribution available on a
    shared key).

    ``settle_seconds`` accounts for OpenRouter's activity pipeline lag (a
    few seconds between completion and ledger update). 5s is conservative
    but reliable.
    """
    if not provisioning_key:
        state = TrialKeyState(key=fallback_key, mode="shared_key")
        yield state
        return

    try:
        minted = await provision_subkey(
            provisioning_key, label=label, limit_usd=limit_usd
        )
    except Exception:
        # Fall back gracefully — don't fail the trial because billing
        # isolation broke.
        state = TrialKeyState(key=fallback_key, mode="shared_key")
        yield state
        return

    state = TrialKeyState(
        key=minted["key"],
        mode="isolated_subkey",
        hash=minted["hash"],
    )
    try:
        yield state
    finally:
        if settle_seconds > 0:
            try:
                await asyncio.sleep(settle_seconds)
            except Exception:
                pass
        try:
            state.usage_usd = await subkey_usage(provisioning_key, state.hash or "")
        except Exception:
            state.usage_usd = None
            state.mode = "isolated_subkey_query_failed"
        if state.hash:
            await delete_subkey(provisioning_key, state.hash)
