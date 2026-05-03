"""Fetch eval ``trace.jsonl`` files from the private HF dataset.

Each eval container is built from ``evals/<slug>/environment/`` and the
Dockerfile copies ``workdir/`` into ``/workdir/``. The trace input lives at
``evals/<slug>/environment/workdir/trace.jsonl`` — this script downloads it
into that exact path so ``harbor run`` (and any plain ``docker build``)
picks it up unchanged.

Source dataset: ``orinlabs/horizon-1-eval-traces`` (private)

Layout on HF:
    <slug>/trace.jsonl              <- what this script fetches
    raw/<slug>/<entity>.raw.jsonl   <- only needed to rebuild trace via
                                       evals/<slug>/scripts/build_trace.py;
                                       opt in with --raw

Auth:
    Set ``HF_TOKEN`` (or run ``huggingface-cli login``) with read access to
    the private dataset.

Examples:
    # All evals (trace only — typical case before harbor run)
    uv run --with huggingface_hub python scripts/fetch_eval_data.py

    # One eval
    uv run --with huggingface_hub python scripts/fetch_eval_data.py \\
        01-direct-semantic-holiday-party-caterer

    # Also pull raw inputs (only needed to re-run build_trace.py)
    uv run --with huggingface_hub python scripts/fetch_eval_data.py --raw

    # Force re-download even if files already exist locally
    uv run --with huggingface_hub python scripts/fetch_eval_data.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "orinlabs/horizon-1-eval-traces"
REPO_TYPE = "dataset"

ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = ROOT / "evals"


def discover_local_slugs() -> list[str]:
    return sorted(p.name for p in EVALS_DIR.iterdir() if p.is_dir())


def fetch_trace(slug: str, *, force: bool) -> Path | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    dest_dir = EVALS_DIR / slug / "environment" / "workdir"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "trace.jsonl"

    if dest.exists() and not force:
        print(f"  trace -> {dest.relative_to(ROOT)} (exists, skip; use --force to refresh)")
        return dest

    try:
        cached = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=f"{slug}/trace.jsonl",
        )
    except EntryNotFoundError:
        print(f"  trace -> (missing on hub: {slug}/trace.jsonl)", file=sys.stderr)
        return None

    dest.write_bytes(Path(cached).read_bytes())
    print(f"  trace -> {dest.relative_to(ROOT)}")
    return dest


def fetch_raws(api, slug: str, *, force: bool) -> list[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
    prefix = f"raw/{slug}/"
    matching = [f for f in files if f.startswith(prefix)]

    if not matching:
        print(f"  raw   -> (no raw files on hub for {slug})")
        return []

    dest_dir = EVALS_DIR / slug / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for hub_path in matching:
        dest = dest_dir / Path(hub_path).name
        if dest.exists() and not force:
            print(f"  raw   -> {dest.relative_to(ROOT)} (exists, skip)")
            written.append(dest)
            continue
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=hub_path,
            )
        except EntryNotFoundError:
            print(f"  raw   -> (missing on hub: {hub_path})", file=sys.stderr)
            continue
        dest.write_bytes(Path(cached).read_bytes())
        print(f"  raw   -> {dest.relative_to(ROOT)}")
        written.append(dest)

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "slugs",
        nargs="*",
        help="Eval slugs to fetch (default: all evals in ./evals/)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also fetch *.raw.jsonl source files (only needed to rebuild traces)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists",
    )
    args = parser.parse_args()

    if not (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or (Path.home() / ".cache/huggingface/token").exists()
    ):
        print(
            "warning: no HF token found (HF_TOKEN / huggingface-cli login). "
            "Private dataset access will fail.",
            file=sys.stderr,
        )

    from huggingface_hub import HfApi

    api = HfApi() if args.raw else None

    slugs = args.slugs or discover_local_slugs()
    if not slugs:
        print("no eval slugs to fetch", file=sys.stderr)
        return 1

    failed = 0
    for slug in slugs:
        if not (EVALS_DIR / slug).is_dir():
            print(f"skip: evals/{slug} not found", file=sys.stderr)
            failed += 1
            continue
        print(f"\n[{slug}]")
        if fetch_trace(slug, force=args.force) is None:
            failed += 1
        if args.raw:
            fetch_raws(api, slug, force=args.force)

    print(f"\ndone ({failed} missing)" if failed else "\ndone")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
