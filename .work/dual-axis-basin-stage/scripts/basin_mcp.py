"""Absolute-path launcher for MCP clients, independent of their working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basin.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "mcp"]))
