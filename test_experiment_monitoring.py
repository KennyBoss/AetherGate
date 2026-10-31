"""Validate real engine telemetry and the exporter/Prometheus/Grafana pipeline."""

import base64
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from prometheus_client import generate_latest, start_http_server
from prometheus_client.parser import text_string_to_metric_families

from experiment_metrics import read_snapshot, registry_for
from research.evolution.telemetry import write_exit_marker
from research.evolution import evolve_ssm_formulas as formulas
from research.evolution import coevolve_ssm_selfplay as selfplay


ROOT = Path(__file__).resolve().parent


def run_case(module, directory, failure=None, seed=None):
    directory = Path(directory)
    flags = ["test", "--agents=4", "--steps=2", f"--output-json={directory / 'result.json'}"]
    if seed is not None:
        flags.append(f"--seed={seed}")
    sigma = "0" if failure == "diversity_collapse" else "3e38" if failure == "nonfinite" else "0.08"
    if module is formulas:
        flags += ["--architecture-population=2", "--architecture-elites=1", "--architecture-generations=0",
                  "--weight-population=2", "--weight-elites=1", "--weight-generations=1", f"--weight-mutation-sigma={sigma}"]
    else:
        flags += ["--runners=2", "--blockers=2", "--runner-elites=1", "--blocker-elites=1",
                  "--generations=1", f"--mutation-sigma={sigma}"]
    with patch.object(sys, "argv", flags):
        args = module.parse_args()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if failure:
            try:
                module.run(args)
            except SystemExit as stopped:
                if stopped.code != 2:
                    raise
            else:
                raise AssertionError("Guardrail did not stop the experiment")
        else:
            module.run(args)
    return read_snapshot(directory / "metrics.json")


def samples(registry):
    text = generate_latest(registry).decode()
    return [sample for family in text_string_to_metric_families(text) for sample in family.samples]


class MonitoringTests(unittest.TestCase):
    def test_real_engines_publish_rewards_entropy_diversity_and_both_stop_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module, kind in ((formulas, "formulas"), (selfplay, "selfplay")):
                for failure in (None, "diversity_collapse", "nonfinite"):
                    snapshot = run_case(module, root / f"{kind}-{failure or 'healthy'}", failure)
                    self.assertEqual(snapshot.status, "stopped" if failure else "completed")
                    if failure:
                        self.assertEqual(snapshot.stop.code, failure)
                        self.assertEqual(snapshot.stop.stage, "after_mutation")
                    for values in snapshot.roles.values():
                        self.assertTrue(math.isfinite(values.reward_mean))
                        self.assertTrue(0 <= values.entropy <= 1.000001)
                        self.assertIsNotNone(values.diversity)
                    if not failure:
                        self.assertEqual(set(snapshot.roles), {"weights"} if kind == "formulas" else {"runner", "blocker"})
            exported = samples(registry_for(root))
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_runs_exported"), 6)
            for reason in ("diversity_collapse", "nonfinite"):
                stopped = [sample for sample in exported if sample.name == "aethergate_run_stop_info" and sample.labels["reason"] == reason]
                self.assertEqual(len(stopped), 2)
            self.assertTrue(all(math.isfinite(sample.value) for sample in exported))

    def test_http_endpoint_refreshes_and_invalid_or_linked_files_are_not_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_case(selfplay, root / "healthy")
            registry = registry_for(root)
            server, thread = start_http_server(0, addr="127.0.0.1", registry=registry)
            try:
                with urlopen(f"http://127.0.0.1:{server.server_port}/metrics", timeout=5) as response:
                    self.assertIn(b"aethergate_policy_entropy_ratio", response.read())
                (root / "broken").mkdir()
                (root / "broken/metrics.json").write_text('{"invalid":NaN}')
                (root / "linked").mkdir()
                (root / "linked/metrics.json").symlink_to(root / "healthy/metrics.json")
                exported = samples(registry)
                self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_files_invalid"), 2)
                self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_runs_exported"), 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_launcher_exit_markers_report_kill_and_timeout_without_overriding_guardrails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_case(selfplay, root / "killed")
            path = root / "killed/metrics.json"
            original = json.loads(path.read_text())
            write_exit_marker(path.parent, 1)
            exported = samples(registry_for(root))
            self.assertTrue(any(sample.name == "aethergate_run_status" and sample.labels["status"] == "failed" and sample.value == 1 for sample in exported))
            original.update(status="running", stop=None)
            for code, reason in ((137, "killed"), (124, "timeout"), (143, "interrupted")):
                path.write_text(json.dumps(original))
                write_exit_marker(path.parent, code)
                exported = samples(registry_for(root))
                self.assertTrue(any(sample.name == "aethergate_run_stop_info" and sample.labels["reason"] == reason for sample in exported))
            run_case(formulas, root / "guarded", "diversity_collapse")
            write_exit_marker(root / "guarded", 2)
            exported = samples(registry_for(root))
            self.assertTrue(any(sample.name == "aethergate_run_stop_info" and sample.labels["reason"] == "diversity_collapse" for sample in exported))

    def test_max_runs_bounds_exposed_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_case(formulas, root / "one")
            run_case(selfplay, root / "two")
            exported = samples(registry_for(root, max_runs=1))
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_runs_exported"), 1)
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_runs_omitted"), 1)


@unittest.skipUnless(os.environ.get("RUN_MONITORING_DOCKER") == "1", "Requires Docker Compose monitoring stack")
class MonitoringStackTests(unittest.TestCase):
    def test_prometheus_scrapes_real_experiments_and_grafana_provisions_dashboard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for module, kind in ((formulas, "formulas"), (selfplay, "selfplay")):
                run_case(module, root / f"{kind}-healthy")
                run_case(module, root / f"{kind}-collapse", "diversity_collapse")
                run_case(module, root / f"{kind}-nonfinite", "nonfinite")
            env = {**os.environ, "MONITORING_ARTIFACT_ROOT": str(root), "MONITORING_UID": str(os.getuid()),
                   "MONITORING_GID": str(os.getgid()), "GRAFANA_ADMIN_PASSWORD": "ci-grafana-password",
                   "METRICS_PORT": "19108", "PROMETHEUS_PORT": "19090", "GRAFANA_PORT": "13000"}
            compose = ["docker", "compose", "-p", f"aethergate-monitor-{os.getpid()}", "-f", str(ROOT / "monitoring/compose.yaml")]
            auth = base64.b64encode(b"admin:ci-grafana-password").decode()
            def get(url, grafana=False):
                headers = {"Authorization": f"Basic {auth}"} if grafana else {}
                with urlopen(Request(url, headers=headers), timeout=5) as response:
                    return json.loads(response.read())
            def wait(check, description):
                deadline = time.monotonic() + 90
                last_error = None
                while time.monotonic() < deadline:
                    try:
                        value = check()
                        if value:
                            return value
                    except (URLError, OSError, ValueError, KeyError, AssertionError) as error:
                        last_error = error
                    time.sleep(.5)
                self.fail(f"Timed out: {description}; last error: {last_error}")
            try:
                subprocess.run([*compose, "up", "--build", "-d"], cwd=ROOT, env=env, check=True, timeout=300)
                def query(expression):
                    return get("http://127.0.0.1:19090/api/v1/query?" + urlencode({"query": expression}))["data"]["result"]
                wait(lambda: query('up{job="aethergate"}') and query('up{job="aethergate"}')[0]["value"][1] == "1", "Prometheus scrape target")
                rewards = wait(lambda: query("aethergate_population_reward"), "population rewards")
                self.assertEqual({item["metric"]["role"] for item in rewards}, {"weights", "runner", "blocker"})
                self.assertEqual(len(query("aethergate_policy_entropy_ratio")), 9)
                self.assertEqual(len(query("aethergate_population_diversity")), 9)
                stops = query("aethergate_run_stop_info")
                self.assertEqual({item["metric"]["reason"] for item in stops}, {"nonfinite", "diversity_collapse"})
                self.assertEqual(query("aethergate_metrics_files_invalid")[0]["value"][1], "0")
                live_query = 'aethergate_population_reward{run="selfplay-healthy",role="runner",stat="mean"}'
                previous_reward = query(live_query)[0]["value"][1]
                run_case(selfplay, root / "selfplay-healthy", seed=23)
                wait(lambda: query(live_query) and query(live_query)[0]["value"][1] != previous_reward,
                     "Prometheus refresh after a new real evaluation")
                dashboard = wait(lambda: get("http://127.0.0.1:13000/api/dashboards/uid/aethergate-evolution", True), "Grafana dashboard provisioning")
                self.assertEqual(dashboard["dashboard"]["uid"], "aethergate-evolution")
                self.assertEqual(len(dashboard["dashboard"]["panels"]), 10)
                for panel in dashboard["dashboard"]["panels"]:
                    for target in panel["targets"]:
                        expression = target["expr"].replace("$experiment", ".*").replace("$run", ".*")
                        self.assertTrue(query(expression), f"Dashboard query has no data: {expression}")
                proxy = wait(lambda: get("http://127.0.0.1:13000/api/datasources/proxy/uid/aethergate-prometheus/api/v1/query?" +
                    urlencode({"query": "aethergate_population_reward"}), True), "Grafana datasource proxy")
                self.assertTrue(proxy["data"]["result"])
                print("PASS monitoring: Prometheus scraped rewards/entropy/diversity and both stop reasons from real runs, observed a live reward update; Grafana dashboard and datasource queries succeeded", flush=True)
            finally:
                subprocess.run([*compose, "logs", "--no-color", "--tail=30"], env=env, check=False, timeout=30)
                subprocess.run([*compose, "down", "--volumes", "--remove-orphans"], env=env, check=False, timeout=60)


if __name__ == "__main__":
    unittest.main()
