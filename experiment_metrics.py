"""Read-only Prometheus exporter for persisted experiment telemetry."""

import argparse
import json
import os
from pathlib import Path
import stat
import threading
from typing import Literal

from prometheus_client import CollectorRegistry, start_http_server
from prometheus_client.core import GaugeMetricFamily
from pydantic import BaseModel, ConfigDict, Field

from research.evolution.telemetry import POPULATIONS, ROLES, STAGES, STATUSES, STOP_CODES


class NumericModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RoleMetrics(NumericModel):
    reward_best: float
    reward_mean: float
    entropy: float
    diversity: float | None


class Stop(NumericModel):
    code: Literal[*STOP_CODES]
    stage: Literal[*STAGES]
    population: Literal[*POPULATIONS]


class Snapshot(NumericModel):
    version: Literal[1]
    experiment: Literal["formulas", "selfplay"]
    git_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|unknown)$")
    status: Literal[*STATUSES]
    heartbeat_enabled: bool = False
    generation: int = Field(ge=0)
    architecture_generation: int = Field(ge=0)
    architecture_index: int = Field(ge=0)
    threshold: float | None
    roles: dict[Literal[*ROLES], RoleMetrics]
    stop: Stop | None
    updated_at: float = Field(gt=0)


def read_json(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Metrics must be a regular file")
        content = stream.read(128 * 1024 + 1)
        if len(content) > 128 * 1024:
            raise ValueError("Metrics snapshot is too large")
    return content


class Heartbeat(NumericModel):
    version: Literal[1]
    updated_at: float = Field(gt=0)


class ExitMarker(NumericModel):
    version: Literal[1]
    exit_code: int = Field(ge=-255, le=255)
    updated_at: float = Field(gt=0)


def read_snapshot(path):
    return Snapshot.model_validate_json(read_json(path))


def with_exit_marker(snapshot, path):
    try:
        marker = ExitMarker.model_validate_json(read_json(path))
    except FileNotFoundError:
        return snapshot
    if marker.updated_at < snapshot.updated_at or snapshot.status == "stopped":
        return snapshot
    code = marker.exit_code
    if snapshot.status != "running" and (code == 0 or snapshot.stop is not None):
        return snapshot
    status = "completed" if code == 0 else "timed_out" if code == 124 else "interrupted" if code in (130, 143) else "failed"
    reason = "timeout" if code == 124 else "killed" if code in (-9, 137) else "interrupted" if code in (130, 143) else "error"
    return snapshot.model_copy(update={"status": status, "updated_at": marker.updated_at,
        "stop": None if code == 0 else Stop(code=reason, stage="runtime", population="unknown")})


class ExperimentCollector:
    def __init__(self, root, max_runs=100):
        self.root = Path(root).resolve()
        self.max_runs = max_runs

    def collect(self):
        labels = ["experiment", "run", "git_commit"]
        metrics = {
            "reward": GaugeMetricFamily("aethergate_population_reward", "Current population reward", labels=labels + ["role", "stat"]),
            "entropy": GaugeMetricFamily("aethergate_policy_entropy_ratio", "Mean normalized action entropy", labels=labels + ["role"]),
            "diversity": GaugeMetricFamily("aethergate_population_diversity", "RMS genome weight standard deviation", labels=labels + ["role"]),
            "threshold": GaugeMetricFamily("aethergate_diversity_threshold", "Diversity stop threshold", labels=labels),
            "generation": GaugeMetricFamily("aethergate_generation", "Current evaluated or stopped generation", labels=labels),
            "architecture_generation": GaugeMetricFamily("aethergate_architecture_generation", "Architecture generation", labels=labels),
            "architecture_index": GaugeMetricFamily("aethergate_architecture_index", "Current architecture index", labels=labels),
            "status": GaugeMetricFamily("aethergate_run_status", "Current run status (one-hot)", labels=labels + ["status"]),
            "stop": GaugeMetricFamily("aethergate_run_stop_info", "Latest stop reason per visible run (not an event counter)", labels=labels + ["reason", "stage", "population"]),
            "heartbeat": GaugeMetricFamily("aethergate_run_heartbeat_timestamp_seconds", "Last process heartbeat; zero means enabled but missing/invalid", labels=labels),
            "updated": GaugeMetricFamily("aethergate_run_updated_timestamp_seconds", "Last successful snapshot timestamp", labels=labels),
        }
        candidates, invalid = [], 0
        for directory, subdirectories, files in os.walk(self.root, followlinks=False):
            subdirectories[:] = [name for name in subdirectories if not (Path(directory) / name).is_symlink()]
            if "metrics.json" in files:
                path = Path(directory) / "metrics.json"
                try:
                    modified = path.stat(follow_symlinks=False).st_mtime_ns
                    if "run_heartbeat.json" in files:
                        try:
                            modified = max(modified, path.with_name("run_heartbeat.json").stat(follow_symlinks=False).st_mtime_ns)
                        except FileNotFoundError:
                            pass
                    candidates.append((modified, path))
                except OSError:
                    invalid += 1
        exported = 0
        for _, path in sorted(candidates, key=lambda item: (-item[0], str(item[1])))[:self.max_runs]:
            try:
                snapshot = with_exit_marker(read_snapshot(path), path.with_name("run_exit.json"))
            except (OSError, ValueError):
                invalid += 1
                continue
            run = path.parent.relative_to(self.root).as_posix()
            common = [snapshot.experiment, run, snapshot.git_commit]
            for role, values in snapshot.roles.items():
                role_labels = common + [role]
                metrics["reward"].add_metric(role_labels + ["best"], values.reward_best)
                metrics["reward"].add_metric(role_labels + ["mean"], values.reward_mean)
                metrics["entropy"].add_metric(role_labels, values.entropy)
                if values.diversity is not None:
                    metrics["diversity"].add_metric(role_labels, values.diversity)
            if snapshot.threshold is not None:
                metrics["threshold"].add_metric(common, snapshot.threshold)
            for key in ("generation", "architecture_generation", "architecture_index"):
                metrics[key].add_metric(common, getattr(snapshot, key))
            for status in STATUSES:
                metrics["status"].add_metric(common + [status], int(snapshot.status == status))
            metrics["updated"].add_metric(common, snapshot.updated_at)
            if snapshot.heartbeat_enabled:
                try:
                    heartbeat = Heartbeat.model_validate_json(read_json(path.with_name("run_heartbeat.json")))
                    timestamp = heartbeat.updated_at
                except (OSError, ValueError):
                    timestamp = 0
                    invalid += 1
                metrics["heartbeat"].add_metric(common, timestamp)
            if snapshot.stop is not None:
                reason = snapshot.stop
                metrics["stop"].add_metric(common + [reason.code, reason.stage, reason.population], 1)
            exported += 1
        yield from metrics.values()
        yield GaugeMetricFamily("aethergate_metrics_files_invalid", "Unreadable or malformed selected metrics files", value=invalid)
        yield GaugeMetricFamily("aethergate_metrics_runs_exported", "Number of selected valid runs", value=exported)
        yield GaugeMetricFamily("aethergate_metrics_runs_omitted", "Older runs omitted by max-runs limit", value=max(0, len(candidates) - self.max_runs))


def registry_for(root, max_runs=100):
    registry = CollectorRegistry()
    registry.register(ExperimentCollector(root, max_runs))
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9108)
    parser.add_argument("--max-runs", type=int, default=100)
    args = parser.parse_args()
    if args.max_runs < 1:
        parser.error("--max-runs must be positive")
    if not args.root.is_dir():
        parser.error("--root must be an existing artifact directory")
    server, thread = start_http_server(args.port, addr=args.listen, registry=registry_for(args.root, args.max_runs))
    print(f"Experiment metrics: http://{args.listen}:{args.port}/metrics", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
