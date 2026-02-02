"""
AGENT RUNNER
============
Runs agents on a node, talks to local LLMs, executes tasks in parallel.

One-liner idea:
    "Agents are actors; tasks are their footsteps."
"""

from __future__ import annotations
import argparse
import os
import time
import requests
from services.agent_runner.worker_pool import WorkerPool
from services.agent_runner.llm_adapters.local_stub import LocalStubLLM

ORCH = os.getenv("ORCH_URL", "http://127.0.0.1:8000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="runner id, e.g. agent_runner@boxA/keeper_01")
    args = ap.parse_args()

    runner_id = args.id
    pool = WorkerPool(max_workers=4)
    llm = LocalStubLLM()

    # Demo loop: heartbeat + placeholder task
    while True:
        # TODO: poll orchestrator for tasks, or subscribe via NATS/WS
        payload = {"status": "ok", "runner": runner_id}
        # (intentionally no network call yet)
        # print heartbeat
        print(f"[heartbeat] {payload}")
        time.sleep(5)


if __name__ == "__main__":
    main()
