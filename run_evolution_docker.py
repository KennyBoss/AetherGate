#!/usr/bin/env python3
"""Run one offline CPU experiment in a restricted, disposable Docker container."""

import argparse
from contextlib import ExitStack
import json
import os
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid


DEFAULT_IMAGE = "aethergate-evolution:local"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("hypothesis", "formulas", "selfplay"), default="hypothesis")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--output-dir", type=Path)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume-from", type=Path)
    resume.add_argument("--resume-s3", help="Download and verify a checkpoint using its S3 manifest URI.")
    parser.add_argument("--s3-uri", help="Publish artifacts under s3://bucket/prefix/<Git SHA>/<run ID>/.")
    parser.add_argument("--s3-endpoint", default=None, help="Custom S3-compatible endpoint; credentials stay on the host.")
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
    if (args.resume_from or args.resume_s3) and args.experiment == "hypothesis":
        parser.error("--resume-from is supported for formulas and selfplay")
    if args.s3_endpoint and not (args.s3_uri or args.resume_s3):
        parser.error("--s3-endpoint requires --s3-uri or --resume-s3")
    if args.s3_uri:
        paths = {"output_json", "checkpoint_path", "feedback_jsonl", "prompt_json", "metrics_path"}
        if any(override.split("=", 1)[0].removeprefix("experiment.parameters.") in paths for override in args.overrides):
            parser.error("S3 publication uses the default artifact filenames inside --output-dir")
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


def remove_container(name):
    # Killing the Docker client alone would leave the experiment running.
    try:
        subprocess.run(["docker", "rm", "--force", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Could not confirm container cleanup for {name}: {exc}", file=sys.stderr)


def image_provenance(image):
    result = subprocess.run(["docker", "image", "inspect", "--format", "{{json .}}", image],
                            check=True, capture_output=True, text=True, timeout=30)
    info = json.loads(result.stdout)
    commit = (info["Config"].get("Labels") or {}).get("org.opencontainers.image.revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("S3 requires an image tagged with its Git commit; rebuild using make docker-build")
    return commit, info["Id"]


def run_archived(command, name, output, timeout, archive, record_exit=False):
    process, reader, error = None, None, None
    code = 1
    reader_errors = []
    (output / "console.log").touch()
    try:
        archive.publish(output, "running")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        def tee():
            try:
                with process.stdout, (output / "console.log").open("a", encoding="utf-8") as log:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        try:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                        except OSError:
                            pass  # Continue preserving logs when the terminal pipe closes.
            except OSError as exc:
                reader_errors.append(exc)
        reader = threading.Thread(target=tee, daemon=True)
        reader.start()
        deadline = time.monotonic() + timeout
        previous = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                code = 124
                break
            try:
                code = process.wait(timeout=min(0.25, remaining))
                break
            except subprocess.TimeoutExpired:
                checkpoint = output / "checkpoint.npz"
                try:
                    info = checkpoint.stat(follow_symlinks=False)
                    signature = (info.st_ino, info.st_mtime_ns, info.st_size)
                except FileNotFoundError:
                    continue
                if signature != previous:
                    archive.publish(output, "running")
                    previous = signature
    except BaseException as exc:
        error = exc
        code = exc.code if isinstance(exc, SystemExit) and isinstance(exc.code, int) else 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        remove_container(name)
        if process is not None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30)
        if reader is not None:
            reader.join(timeout=30)
            if error is None and (reader.is_alive() or reader_errors):
                error = reader_errors[0] if reader_errors else RuntimeError("Console log capture did not finish")
                code = 1
        if record_exit:
            from research.evolution.telemetry import write_exit_marker
            write_exit_marker(output, code)
        status = {0: "completed", 2: "stopped", 124: "timed_out", 130: "interrupted", 143: "interrupted"}.get(code, "failed")
        try:
            archive.publish(output, status, code)
        except Exception as exc:
            if error is None:
                error = exc
            else:
                print(f"Final S3 publication failed; local artifacts remain in {output}: {exc}", file=sys.stderr)
            if code == 0 and record_exit:
                from research.evolution.telemetry import write_exit_marker
                write_exit_marker(output, 1)
    if error is not None:
        raise error
    return code


def run(args):
    if not shutil.which("docker"):
        raise ValueError("Docker CLI was not found; install Docker and start its engine first")
    name = f"aethergate-evolution-{uuid.uuid4().hex}"
    output = mount_path(args.output_dir or Path("artifacts/evolution-docker") / name)
    with ExitStack() as stack:
        archive = None
        if args.s3_uri or args.resume_s3:
            from experiment_artifacts import ArtifactStore, download_checkpoint, s3_client
            commit, image_id = image_provenance(args.image)
            client = s3_client(args.s3_endpoint)
            if args.resume_s3:
                temporary = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aethergate-resume-")))
                args.resume_from = temporary / "checkpoint.npz"
                download_checkpoint(client, args.resume_s3, args.resume_from, commit)
            if args.s3_uri:
                archive = ArtifactStore(client, args.s3_uri, commit, args.experiment, image_id,
                                        run_id=name.rsplit("-", 1)[1], resumed_from=args.resume_s3)
        command = docker_command(args, output, name)
        output.mkdir(parents=True, exist_ok=False)
        print(f"Results: {output}", flush=True)
        code = 1
        try:
            if archive is not None:
                print(f"S3 manifest: {archive.uri}", flush=True)
                code = run_archived(command, name, output, args.timeout, archive, record_exit=True)
            else:
                code = subprocess.run(command, timeout=args.timeout, check=False).returncode
            return code
        except subprocess.TimeoutExpired:
            print(f"Experiment exceeded {args.timeout} seconds; removing container", file=sys.stderr)
            code = 124
            return code
        except BaseException as error:
            code = error.code if isinstance(error, SystemExit) and isinstance(error.code, int) else 130 if isinstance(error, KeyboardInterrupt) else 1
            raise
        finally:
            remove_container(name)
            if archive is None:
                from research.evolution.telemetry import write_exit_marker
                write_exit_marker(output, code)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        raise SystemExit(run(parse_args()))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
