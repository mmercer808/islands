"""
bootstrap.py
============
Create the repo structure and stubs (cross-platform).

Usage:
  python bootstrap.py --path ./island-engine
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path

# NOTE: This script is intentionally minimal; the repo already contains the scaffold.
# It is here so you can regenerate the structure in a fresh location without PowerShell pain.

TREE = [
  "docs",
  "protocol",
  "services/orchestrator",
  "services/agent_runner/llm_adapters",
  "services/agent_runner/sandbox",
  "services/signbook_server",
  "world/canon",
  "world/timelines",
  "world/entities/characters",
  "world/entities/locations",
  "world/entities/items",
  "keepers/procedures",
  "keepers/npc_sheets",
  "agents/templates",
  "agents/personalities",
  "tests",
  "var",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="island-engine")
    args = ap.parse_args()

    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)

    for d in TREE:
        (root / d).mkdir(parents=True, exist_ok=True)

    print(f"Created scaffold at: {root.resolve()}")

if __name__ == "__main__":
    main()
