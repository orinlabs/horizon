"""Environment subclasses that hydrate a Horizon eval's ``trace.jsonl``.

Each eval container contains only static app/tools scaffolding. After the
underlying Harbor environment finishes ``start()`` (image built, sandbox up),
this subclass downloads ``<eval-slug>/trace.jsonl`` from a Hugging Face dataset
on the host and uploads it into the container at ``/workdir/trace.jsonl``. Once
``start()`` returns, the environment is fully ready and the agent can be dropped
in unchanged.

Wire it into ``harbor run`` with::

    --environment-import-path horizon_environment:HorizonModalEnvironment
    # or for local docker:
    --environment-import-path horizon_environment:HorizonDockerEnvironment

(Requires ``PYTHONPATH=agents`` so the module is importable.)
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.modal import ModalEnvironment

TRACE_DATASET_REPO_ID = "orinlabs/horizon-1-example-traces"
TRACE_REMOTE_PATH = "/workdir/trace.jsonl"

_logger = logging.getLogger(__name__)


def _slug_for(environment_dir: Path) -> str:
    """Derive eval slug from ``evals/<slug>/environment/``."""
    return Path(environment_dir).resolve().parent.name


def _trace_dataset_repo_id() -> str:
    return os.environ.get("HORIZON_TRACE_DATASET_REPO_ID") or TRACE_DATASET_REPO_ID


async def _hydrate_trace(env, *, repo_id: str | None = None) -> None:
    """Ensure ``/workdir/trace.jsonl`` is present in the running env.

    Called after the underlying Harbor environment has started. Idempotent:
    no-ops if the file is already there.
    """
    slug = _slug_for(env.environment_dir)

    check = await env.exec(f"test -f {shlex.quote(TRACE_REMOTE_PATH)}", timeout_sec=10)
    if check.return_code == 0:
        _logger.info("[%s] trace already in env, skipping fetch", slug)
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required on the host to fetch eval traces."
        ) from exc

    repo_id = repo_id or _trace_dataset_repo_id()
    _logger.info("[%s] fetching trace.jsonl from HF dataset %s", slug, repo_id)
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=f"{slug}/trace.jsonl",
    )

    parent_dir = TRACE_REMOTE_PATH.rsplit("/", 1)[0] or "/"
    await env.exec(f"mkdir -p {shlex.quote(parent_dir)}", timeout_sec=10)

    _logger.info("[%s] uploading %s -> %s", slug, local_path, TRACE_REMOTE_PATH)
    await env.upload_file(local_path, TRACE_REMOTE_PATH)


class HorizonModalEnvironment(ModalEnvironment):
    """Modal environment that hydrates the eval trace post-start."""

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        await _hydrate_trace(self)


class HorizonDockerEnvironment(DockerEnvironment):
    """Docker environment that hydrates the eval trace post-start."""

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        await _hydrate_trace(self)
