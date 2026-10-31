"""Host-only S3 artifact publication and verified checkpoint download."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Literal
from urllib.parse import unquote, urlsplit
import uuid

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field


FILES = ("checkpoint.npz", "result.json", "console.log", "run_evolution.log", "feedback.jsonl", "prompt.json")
SHA = r"^[0-9a-f]{40}$"
DIGEST = r"^[0-9a-f]{64}$"
TRANSFER = TransferConfig(use_threads=False)


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    key: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=DIGEST)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1] = 1
    git_commit: str = Field(pattern=SHA)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    image_id: str
    experiment: Literal["formulas", "selfplay", "hypothesis"]
    status: Literal["running", "completed", "stopped", "failed", "timed_out", "interrupted"]
    exit_code: int | None
    resumed_from: str | None
    artifacts: dict[str, Artifact]


def parse_uri(uri):
    parsed = urlsplit(uri)
    key = unquote(parsed.path).strip("/")
    if (parsed.scheme != "s3" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or parsed.query or parsed.fragment or any(part in (".", "..") for part in key.split("/"))):
        raise ValueError("Use s3://bucket/prefix without credentials, query or parent path components")
    return parsed.netloc, key


def s3_client(endpoint=None):
    if endpoint:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("S3 endpoint must be an HTTP(S) URL without credentials or query")
    return boto3.client("s3", endpoint_url=endpoint, config=Config(
        signature_version="s3v4", s3={"addressing_style": "path"},
        retries={"mode": "standard", "total_max_attempts": 3}, connect_timeout=5, read_timeout=30,
        request_checksum_calculation="when_required", response_checksum_validation="when_required"))


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def snapshot_file(directory_fd, name, destination):
    """Never follow an agent-created symlink into files on the host."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    except FileNotFoundError:
        return False
    with os.fdopen(descriptor, "rb") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"Artifact must be a regular file: {name}")
        remaining = info.st_size
        with open(destination, "wb") as target:
            while remaining:
                chunk = source.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f"Artifact changed while snapshotting: {name}")
                target.write(chunk)
                remaining -= len(chunk)
    return True


class ArtifactStore:
    def __init__(self, client, uri, git_commit, experiment, image_id, run_id=None, resumed_from=None):
        if not re.fullmatch(SHA, git_commit):
            raise ValueError("S3 publication requires a full Git commit SHA")
        self.client = client
        self.bucket, prefix = parse_uri(uri)
        self.run_id = run_id or uuid.uuid4().hex
        self.root = "/".join(part for part in (prefix, git_commit, self.run_id) if part)
        self.manifest_key = self.root + "/manifest.json"
        self.uri = f"s3://{self.bucket}/{self.manifest_key}"
        self.fields = dict(git_commit=git_commit, run_id=self.run_id, experiment=experiment,
                           image_id=image_id, resumed_from=resumed_from)
        Manifest(**self.fields, status="running", exit_code=None, artifacts={})
        self.uploaded = set()

    def publish(self, output, status, exit_code=None):
        artifacts = {}
        try:
            directory_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                with tempfile.TemporaryDirectory(prefix="aethergate-s3-") as temporary:
                    for name in FILES:
                        snapshot = Path(temporary) / name
                        if not snapshot_file(directory_fd, name, snapshot):
                            continue
                        size, digest = digest_file(snapshot)
                        key = f"{self.root}/objects/{digest}/{name}"
                        if key not in self.uploaded:
                            self.client.upload_file(str(snapshot), self.bucket, key, Config=TRANSFER,
                                ExtraArgs={"Metadata": {"git-commit": self.fields["git_commit"], "sha256": digest}})
                            self.uploaded.add(key)
                        artifacts[name] = Artifact(key=key, size=size, sha256=digest)
            finally:
                os.close(directory_fd)
            if status == "completed" and ("result.json" not in artifacts or (
                    self.fields["experiment"] != "hypothesis" and "checkpoint.npz" not in artifacts)):
                raise ValueError("Completed S3 runs require result.json and, for evolution/self-play, checkpoint.npz")
            manifest = Manifest(**self.fields, status=status, exit_code=exit_code, artifacts=artifacts)
            # Publish the pointer last: failed uploads leave the preceding manifest usable.
            self.client.put_object(Bucket=self.bucket, Key=self.manifest_key,
                                   Body=manifest.model_dump_json().encode(), ContentType="application/json",
                                   Metadata={"git-commit": self.fields["git_commit"]})
            return self.uri
        except (BotoCoreError, ClientError) as error:
            raise RuntimeError(f"S3 upload failed; local artifacts are preserved: {error}") from error


def download_checkpoint(client, uri, destination, expected_commit):
    bucket, key = parse_uri(uri)
    if not key.endswith("/manifest.json"):
        raise ValueError("Resume requires a run's S3 manifest.json URI")
    temporary = None
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        with closing(response["Body"]) as body:
            raw = body.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("S3 manifest is too large")
        manifest = Manifest.model_validate_json(raw)
        if manifest.git_commit != expected_commit:
            raise ValueError("S3 checkpoint Git commit does not match the experiment image")
        root = key.removesuffix("/manifest.json")
        if not root.endswith(f"/{manifest.git_commit}/{manifest.run_id}") and root != f"{manifest.git_commit}/{manifest.run_id}":
            raise ValueError("S3 manifest location does not match its Git commit/run")
        if not set(manifest.artifacts) <= set(FILES):
            raise ValueError("Unexpected S3 artifact name")
        for name, artifact in manifest.artifacts.items():
            if artifact.key != f"{root}/objects/{artifact.sha256}/{name}":
                raise ValueError("S3 artifact points outside its run or digest")
        artifact = manifest.artifacts.get("checkpoint.npz")
        if artifact is None:
            raise ValueError("This S3 manifest has no committed checkpoint")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = stream.name
        client.download_file(bucket, artifact.key, temporary, Config=TRANSFER)
        if digest_file(temporary) != (artifact.size, artifact.sha256):
            raise ValueError("Downloaded checkpoint size/SHA-256 does not match its manifest")
        os.replace(temporary, destination)
        return manifest
    except (BotoCoreError, ClientError) as error:
        raise RuntimeError(f"S3 checkpoint download failed: {error}") from error
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser("upload", help="Retry publication of a locally preserved run")
    upload.add_argument("--output-dir", required=True, type=Path)
    upload.add_argument("--s3-uri", required=True)
    upload.add_argument("--git-commit", required=True)
    upload.add_argument("--experiment", required=True, choices=("formulas", "selfplay", "hypothesis"))
    upload.add_argument("--status", choices=("completed", "failed", "stopped", "interrupted", "timed_out"), required=True)
    download = commands.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--git-commit", required=True)
    download.add_argument("--output-file", required=True, type=Path)
    args = parser.parse_args()
    client = s3_client(args.endpoint)
    if args.command == "upload":
        store = ArtifactStore(client, args.s3_uri, args.git_commit, args.experiment, "manual-upload")
        print(store.publish(args.output_dir, args.status))
    else:
        download_checkpoint(client, args.manifest, args.output_file, args.git_commit)
        print(f"Verified checkpoint: {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
