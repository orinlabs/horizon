"""Environment-owned tool registry support for API agents.

Tasks can publish a tool registry at ``/tools/tools.json`` inside the
environment. The registry is the ownership boundary: it defines the LLM SDK
schemas and the handlers used to execute model tool calls against the
environment's stateful backend.

Agents should treat this module as generic plumbing:

- load the registry from the environment
- pass the advertised ``sdk_schema`` entries directly to the LLM SDK
- route model tool calls back through the declared handlers

This module intentionally has no application-specific logic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from dataclasses import dataclass
from typing import Any

from harbor.environments.base import BaseEnvironment


TOOLS_REGISTRY_PATH = "/tools/tools.json"

# Modal sandboxes occasionally race when many `Image.from_dockerfile` builds
# run concurrently — a sandbox can come up healthy enough to exec, but a
# few of the COPY'd files temporarily appear missing for the first second
# or two. We retry the readiness probe with exponential backoff before
# giving up; the sleeps below sum to ~31s.
_REGISTRY_PROBE_BACKOFFS = (1.0, 2.0, 4.0, 8.0, 16.0)

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvironmentTool:
    name: str
    sdk_schema: dict[str, Any]
    handler: dict[str, Any]


@dataclass(frozen=True)
class EnvironmentToolRegistry:
    schema_version: str
    tools: dict[str, EnvironmentTool]

    @property
    def sdk_schemas(self) -> list[dict[str, Any]]:
        return [tool.sdk_schema for tool in self.tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self.tools


async def load_environment_tool_registry(
    environment: BaseEnvironment,
    *,
    path: str = TOOLS_REGISTRY_PATH,
) -> EnvironmentToolRegistry | None:
    """Load ``/tools/tools.json`` from the task environment if present.

    Returns ``None`` only when the file genuinely does not exist (e.g. a
    legacy task that doesn't publish a registry). Transient sandbox
    readiness failures are retried with backoff before that decision is
    made — see ``_REGISTRY_PROBE_BACKOFFS``.
    """
    raw_text = await _read_registry_file_with_retries(environment, path)
    if raw_text is None:
        return None

    try:
        raw = json.loads(raw_text or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid environment tool registry JSON at {path}: {exc}") from exc

    version = str(raw.get("schema_version") or "")
    if version != "horizon-tools-v1":
        raise RuntimeError(
            f"unsupported environment tool registry schema_version: {version!r}"
        )

    tools: dict[str, EnvironmentTool] = {}
    for item in raw.get("tools") or []:
        name = str(item.get("name") or "")
        sdk_schema = item.get("sdk_schema")
        handler = item.get("handler")
        if not name or not isinstance(sdk_schema, dict) or not isinstance(handler, dict):
            raise RuntimeError(f"invalid tool registry item: {item!r}")
        tools[name] = EnvironmentTool(
            name=name,
            sdk_schema=sdk_schema,
            handler=handler,
        )

    return EnvironmentToolRegistry(schema_version=version, tools=tools)


async def _read_registry_file_with_retries(
    environment: BaseEnvironment,
    path: str,
) -> str | None:
    """Probe + read the registry file, surviving transient sandbox flakes.

    The probe is run several times with exponential backoff. We treat
    *any* failure mode of the underlying exec — non-zero exit code or a
    raised SDK exception (e.g. Modal's "Failed to read exec stdio stream"
    transient) — as a possible flake until we have proven, via a
    successful exec on the parent directory, that the file actually does
    not exist. When we exhaust all attempts we either return ``None``
    (file confirmed missing) or raise ``RuntimeError`` with diagnostics
    (file appeared unreadable).
    """
    parent_dir = path.rsplit("/", 1)[0] or "/"
    last_test_stderr = ""
    last_cat_stderr = ""
    last_ls_parent = ""
    last_exception: BaseException | None = None

    for attempt, backoff_sec in enumerate((0.0, *_REGISTRY_PROBE_BACKOFFS)):
        if backoff_sec:
            await asyncio.sleep(backoff_sec)

        try:
            exists = await environment.exec(
                f"test -f {shlex.quote(path)}",
                timeout_sec=10,
            )
        except Exception as exc:
            last_exception = exc
            _logger.warning(
                "registry probe attempt %d: test -f %s raised %s: %s",
                attempt + 1,
                path,
                type(exc).__name__,
                exc,
            )
            continue

        if exists.return_code == 0:
            try:
                cat = await environment.exec(
                    f"cat {shlex.quote(path)}",
                    timeout_sec=10,
                )
            except Exception as exc:
                last_exception = exc
                _logger.warning(
                    "registry probe attempt %d: cat %s raised %s: %s",
                    attempt + 1,
                    path,
                    type(exc).__name__,
                    exc,
                )
                continue

            if cat.return_code == 0 and (cat.stdout or "").strip():
                return cat.stdout
            last_cat_stderr = (cat.stderr or "").strip() or "<empty>"
            _logger.warning(
                "registry probe attempt %d: cat %s returned code=%d stderr=%r",
                attempt + 1,
                path,
                cat.return_code,
                last_cat_stderr,
            )
            continue

        last_test_stderr = (exists.stderr or "").strip() or "<empty>"

        try:
            ls_parent = await environment.exec(
                f"ls -la {shlex.quote(parent_dir)} 2>&1 || true",
                timeout_sec=10,
            )
            last_ls_parent = (ls_parent.stdout or "").strip()
        except Exception as exc:
            last_exception = exc
            last_ls_parent = f"<ls failed: {type(exc).__name__}: {exc}>"

        _logger.warning(
            "registry probe attempt %d: test -f %s returned code=%d stderr=%r; "
            "ls %s -> %s",
            attempt + 1,
            path,
            exists.return_code,
            last_test_stderr,
            parent_dir,
            last_ls_parent.splitlines()[:8],
        )

    try:
        final_check = await environment.exec(
            f"test -d {shlex.quote(parent_dir)}",
            timeout_sec=10,
        )
    except Exception as exc:
        last_exception = exc
        final_check = None  # type: ignore[assignment]

    if final_check is not None and final_check.return_code != 0:
        # Parent directory does not exist either. Treat as a legacy task
        # that does not publish a tools registry.
        return None

    raise RuntimeError(
        f"failed to read environment tool registry at {path} after "
        f"{len(_REGISTRY_PROBE_BACKOFFS) + 1} attempts. "
        f"last test stderr={last_test_stderr!r}; "
        f"last cat stderr={last_cat_stderr!r}; "
        f"last exception={last_exception!r}; "
        f"ls {parent_dir} ->\n{last_ls_parent}"
    )


def get_environment_tool_schemas(
    registry: EnvironmentToolRegistry | None,
) -> list[dict[str, Any]]:
    """Return SDK-ready schemas from a registry, or an empty list."""
    return registry.sdk_schemas if registry else []


async def call_environment_tool(
    environment: BaseEnvironment,
    registry: EnvironmentToolRegistry,
    name: str,
    args: dict[str, Any],
    *,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """Execute an environment-owned tool call and return a JSONable payload.

    A transient sandbox/stdio failure is retried a couple of times before
    being surfaced to the LLM as a tool error rather than crashing the
    agent loop.
    """
    tool = registry.tools.get(name)
    if tool is None:
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": f"unknown environment tool: {name}",
        }

    handler_type = str(tool.handler.get("type") or "")
    if handler_type != "command":
        return {
            "exit_code": 127,
            "stdout": "",
            "stderr": f"unsupported handler type for {name}: {handler_type}",
        }

    try:
        command = _command_from_handler(tool.handler, args)
    except ValueError as exc:
        return {"exit_code": 2, "stdout": "", "stderr": str(exc)}

    result, error = await resilient_exec(
        environment, command, timeout_sec=timeout_sec
    )
    if result is None:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": (
                f"environment exec for tool {name!r} failed after retries: {error}"
            ),
        }
    return {
        "exit_code": result.return_code,
        "stdout": (result.stdout or "")[-4000:],
        "stderr": (result.stderr or "")[-4000:],
    }


# ~30s of total backoff across 4 retry attempts (5 total tries).
_EXEC_RETRY_BACKOFFS = (1.0, 2.0, 4.0, 8.0)


async def resilient_exec(
    environment: BaseEnvironment,
    command: str,
    *,
    timeout_sec: int | None = None,
):
    """Run ``environment.exec`` with retries on raised exceptions.

    Returns ``(ExecResult, None)`` on success or ``(None, error_str)``
    after all retries are exhausted. Non-zero exit codes are *not*
    treated as retryable — they are real command failures and should be
    surfaced to the caller as the first ExecResult.
    """
    last_error: str = ""
    for attempt, backoff_sec in enumerate((0.0, *_EXEC_RETRY_BACKOFFS)):
        if backoff_sec:
            await asyncio.sleep(backoff_sec)
        try:
            return await environment.exec(command, timeout_sec=timeout_sec), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            _logger.warning(
                "exec attempt %d for command %r raised %s",
                attempt + 1,
                command[:80],
                last_error,
            )
    return None, last_error


def _command_from_handler(handler: dict[str, Any], args: dict[str, Any]) -> str:
    argv = handler.get("argv")
    arg_map = handler.get("arg_map") or {}
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError("command handler requires string argv list")
    if not isinstance(arg_map, dict):
        raise ValueError("command handler arg_map must be an object")

    parts: list[str] = list(argv)
    for key, flag in arg_map.items():
        if key not in args:
            continue
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, bool):
            if value:
                parts.append(str(flag))
            continue
        if isinstance(value, list):
            for item in value:
                parts.extend([str(flag), str(item)])
            continue
        parts.extend([str(flag), str(value)])

    return shlex.join(parts)
