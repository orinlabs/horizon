#!/usr/bin/env python
from __future__ import annotations

import json
import os
import stat
from pathlib import Path


def main() -> int:
    tools_dir = Path(os.environ.get("HORIZON_TOOLS_DIR", "/.horizon/tools"))
    tools_path = tools_dir / "tools.json"
    handler_path = tools_dir / "tool_handler.py"
    bin_dir = Path("/usr/local/bin")

    if not tools_path.is_file():
        raise FileNotFoundError(f"missing tool registry: {tools_path}")
    if not handler_path.is_file():
        raise FileNotFoundError(f"missing tool handler: {handler_path}")

    registry = json.loads(tools_path.read_text())
    for tool in registry.get("tools", []):
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid tool entry: {tool!r}")

        command_path = bin_dir / name
        command_path.write_text(
            f'#!/usr/bin/env bash\nset -euo pipefail\nexec python "{handler_path}" "{name}" "$@"\n'
        )
        command_path.chmod(
            command_path.stat().st_mode
            | stat.S_IXUSR
            | stat.S_IXGRP
            | stat.S_IXOTH
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
