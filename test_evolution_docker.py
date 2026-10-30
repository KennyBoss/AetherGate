"""Launcher regressions plus real Docker checks enabled explicitly by CI."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_evolution_docker as launcher


ROOT = Path(__file__).resolve().parent


class DockerCommandTests(unittest.TestCase):
    def test_restricted_command_and_readonly_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "proposals with spaces.json"
            source.write_text("[]")
            args = launcher.parse_args(["--proposals", str(source), "experiment.parameters.seed=17"])
            with patch.object(os, "getuid", return_value=1001), patch.object(os, "getgid", return_value=1001):
                command = launcher.docker_command(args, Path(directory) / "output", "test-container")
            for flag in ("--user=1001:1001", "--read-only", "--network=none", "--cap-drop=ALL",
                         "--security-opt=no-new-privileges=true", "--pids-limit=256", "--cpus=2",
                         "--memory=2g", "--memory-swap=2g", "--pull=never"):
                self.assertIn(flag, command)
            mounts = [arg for arg in command if arg.startswith("--mount=")]
            self.assertEqual(len(mounts), 2)
            self.assertTrue(mounts[1].endswith("dst=/input/proposals.json,readonly"))
            self.assertIn("experiment.parameters.proposals=/input/proposals.json", command)
            self.assertEqual(command[-1], "hydra.run.dir=/output")
            self.assertNotIn("--privileged", command)

    def test_rejects_root_missing_input_and_invalid_overrides(self):
        args = launcher.parse_args([])
        with patch.object(os, "getuid", return_value=0), self.assertRaises(ValueError):
            launcher.docker_command(args, "/tmp/output", "test-container")
        with self.assertRaises(ValueError):
            launcher.mount_path("/tmp/comma,path")
        with patch.object(os, "getuid", return_value=1001), self.assertRaises(ValueError):
            launcher.docker_command(launcher.parse_args(["--proposals", "/does-not-exist"]),
                                    "/tmp/output", "test-container")
        for argv in (["--timeout", "0"], ["--experiment", "selfplay", "--proposals", "x"],
                     ["hydra.run.dir=/host"], ["--privileged"], ["--image=--privileged"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launcher.parse_args(argv)

    def test_timeout_removes_container_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            args = launcher.parse_args(["--output-dir", str(output), "--timeout", "1"])
            with patch.object(os, "getuid", return_value=1001), \
                 patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"), \
                 patch.object(subprocess, "run", side_effect=[
                     subprocess.TimeoutExpired("docker", 1), subprocess.CompletedProcess([], 0)
                 ]) as run, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(launcher.run(args), 124)
                run_command = run.call_args_list[0].args[0]
                name = next(flag.split("=", 1)[1] for flag in run_command if flag.startswith("--name="))
                self.assertEqual(run.call_args_list[1].args[0], ["docker", "rm", "--force", name])
                self.assertNotIn("shell", run.call_args_list[0].kwargs)
            (output / "keep.txt").write_text("existing result")
            with patch.object(os, "getuid", return_value=1001), \
                 patch.object(launcher.shutil, "which", return_value="/usr/bin/docker"), \
                 patch.object(subprocess, "run") as run, self.assertRaises(FileExistsError):
                launcher.run(args)
            run.assert_not_called()
            self.assertEqual((output / "keep.txt").read_text(), "existing result")


@unittest.skipUnless(os.environ.get("RUN_DOCKER_TESTS") == "1", "Run make docker-test with Docker Engine")
class DockerIntegrationTests(unittest.TestCase):
    def test_real_experiment_modes(self):
        image = os.environ.get("EVOLUTION_DOCKER_IMAGE", launcher.DEFAULT_IMAGE)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposals = root / "proposals.json"
            formula = dict(memory_op="identity", input_op="identity", combine_op="add", final_op="tanh")
            proposals.write_text(json.dumps([
                {"id": "valid", "formula": formula},
                {"id": "invalid", "formula": {**formula, "memory_op": "exec"}},
            ]))
            for kind in ("hypothesis", "formulas", "selfplay"):
                output = root / kind
                command = [sys.executable, str(ROOT / "run_evolution_docker.py"), "--experiment", kind,
                           "--image", image, "--output-dir", str(output), "--timeout", "120"]
                if kind == "hypothesis":
                    command += ["--proposals", str(proposals)]
                command += ["experiment.parameters.agents=4", "experiment.parameters.steps=2"]
                if kind == "selfplay":
                    command += ["experiment.parameters.runners=2", "experiment.parameters.blockers=2",
                                "experiment.parameters.generations=1"]
                else:
                    command += ["experiment.parameters.weight_population=2",
                                "experiment.parameters.weight_elites=1",
                                "experiment.parameters.weight_generations=1"]
                    if kind == "formulas":
                        command += ["experiment.parameters.architecture_population=2",
                                    "experiment.parameters.architecture_elites=1"]
                result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=160)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                payload = json.loads((output / "result.json").read_text())
                self.assertEqual(payload["backend"], "cpu")
                self.assertTrue((output / ".hydra/config.yaml").exists())
                if kind == "hypothesis":
                    self.assertEqual(payload["protocol"]["accepted"], 1)
                    self.assertEqual(payload["protocol"]["rejected"], 1)
                print(f"PASS Docker {kind}: CPU result and saved Hydra config", flush=True)

    def test_runtime_isolation(self):
        image = os.environ.get("EVOLUTION_DOCKER_IMAGE", launcher.DEFAULT_IMAGE)
        probe = """
import json, os
from pathlib import Path
assert os.getuid() != 0
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
assert int(status['CapEff'].strip(), 16) == 0
assert status['NoNewPrivs'].strip() == '1'
assert os.statvfs('/app').f_flag & os.ST_RDONLY
assert os.statvfs('/input/proposals.json').f_flag & os.ST_RDONLY
assert sorted(os.listdir('/sys/class/net')) == ['lo']
assert not Path('/var/run/docker.sock').exists()
assert not Path('/app/.git').exists()
cgroup = Path('/sys/fs/cgroup')
assert (cgroup / 'memory.max').read_text().strip() == '2147483648'
assert (cgroup / 'memory.swap.max').read_text().strip() == '0'
assert (cgroup / 'pids.max').read_text().strip() == '256'
quota, period = map(int, (cgroup / 'cpu.max').read_text().split())
assert quota / period == 2
Path('/output/probe.json').write_text(json.dumps({'isolation': 'passed', 'uid': os.getuid()}))
"""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            proposals = Path(directory) / "proposals.json"
            proposals.write_text("[]")
            args = launcher.parse_args(["--image", image, "--proposals", str(proposals)])
            command = launcher.docker_command(args, output, "aethergate-isolation-probe")
            index = command.index(image)
            command = [*command[:index], "--entrypoint=python", image, "-c", probe]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            finally:
                subprocess.run(["docker", "rm", "--force", "aethergate-isolation-probe"], check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((output / "probe.json").read_text())["isolation"], "passed")
            print("PASS Docker isolation: non-root, read-only, offline, no capabilities, resource limits", flush=True)


if __name__ == "__main__":
    unittest.main()
