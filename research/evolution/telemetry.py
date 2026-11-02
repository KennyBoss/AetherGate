"""Atomic numeric snapshots; networking stays outside experiment containers."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import sys
import time


HEARTBEAT_INTERVAL_SECONDS = 2

STATUSES = ("running", "completed", "stopped", "failed", "interrupted", "timed_out")
ROLES = ("weights", "runner", "blocker")
STOP_CODES = ("nonfinite", "diversity_collapse", "invalid_configuration", "error", "interrupted", "killed", "timeout")
STAGES = ("configuration", "initialization", "warmup", "evaluation", "after_mutation", "runtime")
POPULATIONS = ("weights", "runner", "blocker", "match", "unknown")


def add_metrics_arguments(parser):
    parser.add_argument("--metrics-path", default=None, help="Atomic metrics JSON; defaults beside output JSON.")


def git_revision():
    revision = os.environ.get("AETHERGATE_GIT_COMMIT", "")
    if not revision:
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2], stderr=subprocess.DEVNULL, text=True, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            revision = "unknown"
    return revision if re.fullmatch(r"[0-9a-f]{40}", revision) else "unknown"


def atomic_json(path, data):
    content = json.dumps(data, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_exit_marker(output, code):
    atomic_json(Path(output) / "run_exit.json", {"version": 1, "exit_code": code, "updated_at": time.time()})


class Telemetry:
    def __init__(self, experiment, args):
        self.path = Path(args.metrics_path or (
            Path(args.output_json).with_name("metrics.json") if args.output_json else f"{experiment}.metrics.json"))
        self.data = {"version": 1, "experiment": experiment, "git_commit": git_revision(), "status": "running",
                     "generation": 0, "architecture_generation": 0, "architecture_index": 0,
                     "threshold": None, "roles": {}, "stop": None, "updated_at": time.time(), "heartbeat_enabled": True}
        threshold = args.min_population_diversity
        if math.isfinite(threshold) and threshold > 0:
            self.data["threshold"] = threshold
        self._heartbeat_stop = threading.Event()
        self._write_heartbeat()
        self.write()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _write_heartbeat(self):
        atomic_json(self.path.with_name("run_heartbeat.json"), {"version": 1, "updated_at": time.time()})

    def _heartbeat_loop(self):
        while not self._heartbeat_stop.wait(HEARTBEAT_INTERVAL_SECONDS):
            try:
                self._write_heartbeat()
            except OSError as error:
                print(f"Experiment heartbeat could not be written: {error}", file=sys.stderr, flush=True)
                return

    def write(self):
        self.data["updated_at"] = time.time()
        atomic_json(self.path, self.data)

    def observe(self, role, generation, *, reward_best, reward_mean, entropy, diversity,
                architecture_generation=0, architecture_index=0):
        values = dict(reward_best=reward_best, reward_mean=reward_mean, entropy=entropy, diversity=diversity)
        if role not in ROLES or any(value is not None and not math.isfinite(value) for value in values.values()):
            raise ValueError("Telemetry requires a known role and finite numeric values")
        self.data.update(generation=generation, architecture_generation=architecture_generation,
                         architecture_index=architecture_index)
        self.data["roles"][role] = values
        self.write()

    def checkpoint_state(self):
        return deepcopy({key: self.data[key] for key in ("roles", "generation", "architecture_generation", "architecture_index")})

    def restore(self, state):
        if state is not None:
            self.data.update(deepcopy(state))
            self.write()

    def finish(self, status, reason=None):
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=5)
        self.data["status"] = status
        if reason is not None:
            population = {"runners": "runner", "blockers": "blocker"}.get(reason.get("population"), reason.get("population", "unknown"))
            self.data["stop"] = {"code": reason["code"] if reason["code"] in STOP_CODES else "error",
                                 "stage": reason.get("stage", "runtime"),
                                 "population": population if population in POPULATIONS else "unknown"}
            if reason.get("generation") is not None:
                self.data["generation"] = reason["generation"]
            if reason["code"] == "diversity_collapse" and population in self.data["roles"]:
                self.data["roles"][population]["diversity"] = reason["diversity"]
        self.write()
