"""
make_structure.py
=================
Cross-platform scaffold generator for this repo layout (no PowerShell needed).

Usage:
  python scripts/make_structure.py --path ./island-engine
"""

from __future__ import annotations
import argparse
from pathlib import Path

TREE = [
  ".github/workflows",
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
  "scripts",
  "var",
  "signbook/legacy",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="island-engine")
    args = ap.parse_args()
    root = Path(args.path)
    root.mkdir(parents=True, exist_ok=True)
    for d in TREE:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"Created structure at: {root.resolve()}")

if __name__ == "__main__":
    main()
