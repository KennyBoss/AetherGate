#!/usr/bin/env python3
"""Run one offline CPU experiment in a restricted, disposable Docker container."""

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import uuid


DEFAULT_IMAGE = "aethergate-evolution:local"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("hypothesis", "formulas", "selfplay"), default="hypothesis")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--timeout", type=int, default=600, help="Maximum run time in seconds.")
    parser.add_argument("overrides", nargs="*", help="Hydra experiment.parameters.KEY=VALUE overrides.")
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not args.image or args.image.startswith("-") or any(char.isspace() for char in args.image):
        parser.error("--image must be a local Docker image name")
    for override in args.overrides:
        if not override.startswith("experiment.parameters.") or "=" not in override:
            parser.error("Only experiment.parameters.KEY=VALUE overrides are accepted")
    if args.proposals and args.experiment != "hypothesis":
        parser.error("--proposals is only supported for the hypothesis experiment")
    if args.resume_from and args.experiment == "hypothesis":
        parser.error("--resume-from is supported for formulas and selfplay")
    return args


def mount_path(path):
    path = Path(path).resolve()
    if "," in str(path):
        raise ValueError("Docker bind mount paths must not contain commas")
    return path


def docker_command(args, output, name):
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise ValueError("Run this launcher as a non-root user")
    command = [
        "docker", "run", "--rm", "--init", "--pull=never", f"--name={name}",
        f"--user={uid}:{gid}", "--read-only", "--network=none",
        "--cap-drop=ALL", "--security-opt=no-new-privileges=true",
        "--pids-limit=256", "--cpus=2", "--memory=2g", "--memory-swap=2g",
        "--tmpfs=/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--env=OMP_NUM_THREADS=2", "--env=OPENBLAS_NUM_THREADS=1",
        f"--mount=type=bind,src={mount_path(output)},dst=/output",
    ]
    overrides = list(args.overrides)
    if args.proposals:
        source = mount_path(args.proposals)
        if not source.is_file():
            raise ValueError(f"Proposal file does not exist: {source}")
        command.append(f"--mount=type=bind,src={source},dst=/input/proposals.json,readonly")
        overrides.append("experiment.parameters.proposals=/input/proposals.json")
    if args.resume_from:
        source = mount_path(args.resume_from)
        if not source.is_file():
            raise ValueError(f"Checkpoint file does not exist: {source}")
        command.append(f"--mount=type=bind,src={source},dst=/input/checkpoint.npz,readonly")
        overrides.append("experiment.parameters.resume_from=/input/checkpoint.npz")
    return [*command, args.image, f"experiment={args.experiment}", *overrides, "hydra.run.dir=/output"]


def run(args):
    if not shutil.which("docker"):
        raise ValueError("Docker CLI was not found; install Docker and start its engine first")
    name = f"aethergate-evolution-{uuid.uuid4().hex}"
    output = mount_path(args.output_dir or Path("artifacts/evolution-docker") / name)
    command = docker_command(args, output, name)
    output.mkdir(parents=True, exist_ok=False)
    print(f"Results: {output}", flush=True)
    try:
        return subprocess.run(command, timeout=args.timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        print(f"Experiment exceeded {args.timeout} seconds; removing container", file=sys.stderr)
        return 124
    finally:
        # Killing the Docker client alone would leave the experiment running.
        try:
            subprocess.run(["docker", "rm", "--force", name], check=False, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"Could not confirm container cleanup for {name}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
