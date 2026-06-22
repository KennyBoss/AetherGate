#!/usr/bin/env python3
"""Execute resumable batches from a plan_text_sweep.py JSON plan."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_RE = re.compile(r"run_(\d+)_")
ROOT = Path(__file__).resolve().parent


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def completed_indices(output_dir: Path) -> set[int]:
    done: set[int] = set()
    for experiment in output_dir.glob("run_*/experiment.json"):
        match = RUN_RE.search(experiment.parent.name)
        if match:
            done.add(int(match.group(1)))
    return done


def batch_indices(batch: dict[str, Any]) -> set[int]:
    start = int(batch["start_index"])
    size = int(batch["max_runs"])
    return set(range(start, start + size))


def batch_status(batch: dict[str, Any], done: set[int]) -> str:
    indexes = batch_indices(batch)
    if indexes <= done:
        return "complete"
    if indexes & done:
        return "partial"
    return "pending"


def choose_batches(plan: dict[str, Any], args: argparse.Namespace, done: set[int]) -> list[dict[str, Any]]:
    batches = plan.get("batches", [])
    if args.batch_start is not None:
        batches = [batch for batch in batches if int(batch["start_index"]) >= args.batch_start]
    selected: list[dict[str, Any]] = []
    for batch in batches:
        status = batch_status(batch, done)
        if status == "complete":
            continue
        selected.append(batch)
        if len(selected) >= args.max_batches:
            break
    return selected


def output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def command_timeout(cmd: list[str], batch: dict[str, Any], args: argparse.Namespace) -> int:
    if args.batch_timeout is not None:
        return args.batch_timeout
    if "--timeout" in cmd:
        inner_timeout = int(cmd[cmd.index("--timeout") + 1])
    elif args.timeout is not None:
        inner_timeout = args.timeout
    else:
        inner_timeout = 420
    return inner_timeout * int(batch["max_runs"]) + 60


def run_command(cmd: list[str], timeout: int) -> dict[str, Any]:
    env = os.environ.copy()
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_PLATFORMS", "cpu")
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = output_text(exc.stdout) + output_text(exc.stderr)
        return {
            "command": cmd,
            "returncode": 124,
            "timed_out": True,
            "timeout_s": timeout,
            "elapsed_s": elapsed,
            "stdout_tail": stdout.splitlines()[-40:],
        }
    elapsed = time.perf_counter() - start
    return {
        "command": cmd,
        "returncode": proc.returncode,
        "timed_out": False,
        "timeout_s": timeout,
        "elapsed_s": elapsed,
        "stdout_tail": proc.stdout.splitlines()[-40:],
    }


def progress_report(plan: dict[str, Any], output_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    done = completed_indices(output_dir)
    run_count = int(plan["grid"]["run_count"])
    batches = plan.get("batches", [])
    return {
        "updated_at": timestamp(),
        "plan_output_dir": plan["output_dir"],
        "completed_runs": len(done),
        "total_runs": run_count,
        "completion_rate": len(done) / max(1, run_count),
        "completed_indices": sorted(done),
        "complete_batches": sum(1 for batch in batches if batch_status(batch, done) == "complete"),
        "partial_batches": sum(1 for batch in batches if batch_status(batch, done) == "partial"),
        "pending_batches": sum(1 for batch in batches if batch_status(batch, done) == "pending"),
        "executions": runs,
    }


def run(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan)
    plan = load_json(plan_path)
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    done = completed_indices(output_dir)
    selected = choose_batches(plan, args, done)

    print("TextPy/SoA sweep plan runner")
    print(f"plan: {plan_path}")
    print(f"output_dir: {output_dir}")
    print(f"completed_runs: {len(done)}/{plan['grid']['run_count']}")
    print(f"selected_batches: {len(selected)}")

    executions: list[dict[str, Any]] = []
    for batch in selected:
        cmd = list(batch["command"])
        if args.timeout is not None:
            if "--timeout" in cmd:
                cmd[cmd.index("--timeout") + 1] = str(args.timeout)
            else:
                cmd.extend(["--timeout", str(args.timeout)])
        print("")
        print(f"batch start={batch['start_index']} max_runs={batch['max_runs']}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            executions.append({"batch": batch, "dry_run": True, "command": cmd})
            continue
        timeout = command_timeout(cmd, batch, args)
        result = run_command(cmd, timeout)
        result["batch"] = batch
        executions.append(result)
        if result["returncode"] != 0:
            print("batch failed")
            break

    progress = progress_report(plan, output_dir, executions)
    output_json = Path(args.output_json) if args.output_json else output_dir / "plan_progress.json"
    write_json(output_json, progress)
    print("")
    print(f"completed_runs: {progress['completed_runs']}/{progress['total_runs']}")
    print(f"complete_batches: {progress['complete_batches']}")
    print(f"partial_batches: {progress['partial_batches']}")
    print(f"pending_batches: {progress['pending_batches']}")
    print(f"Saved progress JSON: {output_json}")
    if any(item.get("returncode", 0) != 0 for item in executions):
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run next batches from a text sweep plan.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--batch-start", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument(
        "--batch-timeout",
        type=int,
        default=None,
        help="Wall-clock limit for each launched sweep batch. Defaults to per-run timeout * batch size + 60s.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
