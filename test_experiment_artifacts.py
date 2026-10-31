"""Remote integrity, host-file isolation and publication failure semantics."""

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from experiment_artifacts import ArtifactStore, download_checkpoint, parse_uri, s3_client
import run_evolution_docker as launcher


COMMIT = "a" * 40


class MemoryS3:
    def __init__(self):
        self.objects = {}
        self.metadata = {}
        self.fail_upload = False

    def upload_file(self, filename, bucket, key, **kwargs):
        if self.fail_upload:
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject")
        self.objects[bucket, key] = Path(filename).read_bytes()
        self.metadata[bucket, key] = kwargs["ExtraArgs"]["Metadata"]

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Bucket, Key] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Bucket, Key])}

    def download_file(self, bucket, key, filename, **kwargs):
        Path(filename).write_bytes(self.objects[bucket, key])


class ArtifactTests(unittest.TestCase):
    def store(self, client):
        return ArtifactStore(client, "s3://aethergate-tests/experiments", COMMIT, "formulas", "sha256:image")

    def test_roundtrip_commit_manifest_and_failed_upload_preserve_remote_snapshot(self):
        client = MemoryS3()
        store = self.store(client)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint.npz").write_bytes(b"generation-one")
            (output / "console.log").write_text("generation 1\n")
            store.publish(output, "running")
            previous = client.objects[store.bucket, store.manifest_key]
            manifest = json.loads(previous)
            self.assertIn(f"/{COMMIT}/", store.uri)
            self.assertEqual(manifest["git_commit"], COMMIT)
            (output / "checkpoint.npz").write_bytes(b"generation-two")
            client.fail_upload = True
            with self.assertRaisesRegex(RuntimeError, "local artifacts are preserved"):
                store.publish(output, "running")
            self.assertEqual(previous, client.objects[store.bucket, store.manifest_key])
            download_checkpoint(client, store.uri, output / "downloaded.npz", COMMIT)
            self.assertEqual((output / "downloaded.npz").read_bytes(), b"generation-one")
            self.assertEqual((output / "checkpoint.npz").read_bytes(), b"generation-two")

    def test_corrupt_remote_bytes_and_foreign_commit_cannot_replace_local_checkpoint(self):
        client = MemoryS3()
        store = self.store(client)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint.npz").write_bytes(b"good")
            store.publish(output, "running")
            manifest = json.loads(client.objects[store.bucket, store.manifest_key])
            destination = output / "existing.npz"
            destination.write_bytes(b"previous")
            with self.assertRaisesRegex(ValueError, "Git commit"):
                download_checkpoint(client, store.uri, destination, "b" * 40)
            artifact = manifest["artifacts"]["checkpoint.npz"]
            client.objects[store.bucket, artifact["key"]] = b"corrupt"
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                download_checkpoint(client, store.uri, destination, COMMIT)
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertEqual(sorted(path.name for path in output.iterdir()), ["checkpoint.npz", "existing.npz"])

    def test_manifest_paths_and_host_symlinks_are_rejected(self):
        client = MemoryS3()
        store = self.store(client)
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as secret_directory:
            output = Path(directory)
            secret = Path(secret_directory) / "secret"
            secret.write_text("host credential")
            (output / "checkpoint.npz").symlink_to(secret)
            with self.assertRaises(OSError):
                store.publish(output, "running")
            self.assertFalse(client.objects)
            (output / "checkpoint.npz").unlink()
            (output / "checkpoint.npz").write_bytes(b"checkpoint")
            store.publish(output, "running")
            original = json.loads(client.objects[store.bucket, store.manifest_key])
            bad = deepcopy(original)
            bad["artifacts"]["checkpoint.npz"]["key"] = "other-run/checkpoint.npz"
            client.objects[store.bucket, store.manifest_key] = json.dumps(bad).encode()
            with self.assertRaisesRegex(ValueError, "outside"):
                download_checkpoint(client, store.uri, output / "download.npz", COMMIT)
            bad = deepcopy(original)
            bad["artifacts"]["../secret"] = bad["artifacts"]["checkpoint.npz"]
            client.objects[store.bucket, store.manifest_key] = json.dumps(bad).encode()
            with self.assertRaisesRegex(ValueError, "artifact name"):
                download_checkpoint(client, store.uri, output / "download.npz", COMMIT)

    def test_completed_manifest_requires_results_and_sdk_uses_safe_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "require result.json"):
                self.store(MemoryS3()).publish(Path(directory), "completed", 0)
        for uri in ("https://bucket/key", "s3://bucket/../key", "s3://user:password@bucket/key", "s3://bucket/key?token=x"):
            with self.assertRaises(ValueError):
                parse_uri(uri)
        for endpoint in ("https://user:secret@example.com", "https://example.com?token=secret", "file:///tmp"):
            with self.assertRaises(ValueError):
                s3_client(endpoint)
        with patch("experiment_artifacts.boto3.client") as create:
            s3_client("https://storage.example.com")
        config = create.call_args.kwargs["config"]
        self.assertEqual(config.signature_version, "s3v4")
        self.assertEqual(config.s3["addressing_style"], "path")
        self.assertNotIn("aws_secret_access_key", create.call_args.kwargs)


class ArchivedLauncherTests(unittest.TestCase):
    def test_periodic_checkpoint_publication_and_console_capture(self):
        archive = Mock()
        process = Mock()
        process.stdout = io.StringIO("engine output\nengine error\n")
        process.wait.side_effect = [subprocess.TimeoutExpired("docker", .25), 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "checkpoint.npz").write_bytes(b"committed")
            with patch.object(launcher.subprocess, "Popen", return_value=process), \
                 patch.object(launcher, "remove_container") as remove, patch.object(launcher.sys, "stdout", io.StringIO()):
                code = launcher.run_archived(["docker"], "container", output, 120, archive)
            self.assertEqual(code, 0)
            self.assertEqual((output / "console.log").read_text(), "engine output\nengine error\n")
            self.assertEqual([call.args[1] for call in archive.publish.call_args_list], ["running", "running", "completed"])
            remove.assert_called_once_with("container")

    def test_log_write_failure_cannot_be_reported_as_success(self):
        class BrokenLog(io.StringIO):
            def write(self, value):
                raise OSError("log disk full")
        archive = Mock()
        process = Mock()
        process.stdout = io.StringIO("engine output\n")
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "remove_container"),              patch.object(launcher.subprocess, "Popen", return_value=process),              patch("run_evolution_docker.Path.open", return_value=BrokenLog()):
            with self.assertRaisesRegex(OSError, "log disk full"):
                launcher.run_archived(["docker"], "container", Path(directory), 120, archive)
        self.assertEqual(archive.publish.call_args_list[-1].args[1:], ("failed", 1))

    def test_storage_failure_stops_container_without_reporting_success(self):
        archive = Mock()
        archive.publish.side_effect = [RuntimeError("upload failed"), None]
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, "remove_container") as remove, \
             patch.object(launcher.subprocess, "Popen") as start:
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                launcher.run_archived(["docker"], "container", Path(directory), 1, archive)
            start.assert_not_called()
            remove.assert_called_once_with("container")
            self.assertEqual(archive.publish.call_args_list[-1].args[1:], ("failed", 1))

    def test_missing_image_commit_and_incompatible_flags_are_rejected(self):
        result = subprocess.CompletedProcess([], 0, json.dumps({"Id": "image", "Config": {"Labels": {}}}))
        with patch.object(launcher.subprocess, "run", return_value=result), self.assertRaisesRegex(ValueError, "Git commit"):
            launcher.image_provenance("old-image")
        for flags in (["--resume-from", "local", "--resume-s3", "s3://bucket/manifest.json"],
                      ["--s3-uri", "s3://bucket/runs", "experiment.parameters.checkpoint_path=/tmp/elsewhere"],
                      ["--s3-endpoint", "http://localhost:5000"]):
            with patch.object(launcher.sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(flags)


if __name__ == "__main__":
    unittest.main()
