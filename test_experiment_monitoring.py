"""Validate real engine telemetry and the exporter/Prometheus/Grafana pipeline."""

import base64
import contextlib
import io
import json
import math
from types import SimpleNamespace
import signal
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
from research.evolution.telemetry import Telemetry, write_exit_marker
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
    def test_heartbeat_refreshes_without_generation_progress_and_stops_on_finish(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.evolution.telemetry.HEARTBEAT_INTERVAL_SECONDS", .02):
            root = Path(directory)
            args = SimpleNamespace(metrics_path=str(root / "metrics.json"), output_json=None, min_population_diversity=1e-6)
            telemetry = Telemetry("selfplay", args)
            try:
                original_metrics = (root / "metrics.json").read_bytes()
                original_heartbeat = json.loads((root / "run_heartbeat.json").read_text())["updated_at"]
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if json.loads((root / "run_heartbeat.json").read_text())["updated_at"] > original_heartbeat:
                        break
                    time.sleep(.01)
                self.assertGreater(json.loads((root / "run_heartbeat.json").read_text())["updated_at"], original_heartbeat)
                self.assertEqual((root / "metrics.json").read_bytes(), original_metrics)
            finally:
                telemetry.finish("completed")
            heartbeat = (root / "run_heartbeat.json").read_bytes()
            time.sleep(.06)
            self.assertEqual((root / "run_heartbeat.json").read_bytes(), heartbeat)

    def test_missing_enabled_heartbeat_is_visible_but_legacy_snapshots_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_case(formulas, root / "run")
            heartbeat = root / "run/run_heartbeat.json"
            heartbeat.unlink()
            exported = samples(registry_for(root))
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_run_heartbeat_timestamp_seconds"), 0)
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_files_invalid"), 1)
            path = root / "run/metrics.json"
            snapshot = json.loads(path.read_text())
            del snapshot["heartbeat_enabled"]
            path.write_text(json.dumps(snapshot))
            exported = samples(registry_for(root))
            self.assertFalse(any(sample.name == "aethergate_run_heartbeat_timestamp_seconds" for sample in exported))
            self.assertEqual(next(sample.value for sample in exported if sample.name == "aethergate_metrics_files_invalid"), 0)

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

    def test_live_heartbeat_keeps_a_long_generation_in_the_bounded_selection(self):
        with tempfile.TemporaryDirectory() as directory, patch("research.evolution.telemetry.HEARTBEAT_INTERVAL_SECONDS", .02):
            root = Path(directory)
            args = SimpleNamespace(metrics_path=str(root / "active/metrics.json"), output_json=None, min_population_diversity=1e-6)
            telemetry = Telemetry("formulas", args)
            try:
                run_case(selfplay, root / "newer-completed")
                time.sleep(.06)
                exported = samples(registry_for(root, max_runs=1))
                self.assertEqual({sample.labels["run"] for sample in exported if sample.name == "aethergate_run_status"}, {"active"})
            finally:
                telemetry.finish("completed")

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
            import yaml
            alert_config = yaml.safe_load((ROOT / "monitoring/alertmanager.yml").read_text())
            alert_config["receivers"][0]["webhook_configs"] = [{"url": "http://alert-receiver:8080/alerts", "send_resolved": True}]
            config_path = root / "alertmanager-ci.yml"
            config_path.write_text(yaml.safe_dump(alert_config))
            env = {**os.environ, "ALERTMANAGER_CONFIG": str(config_path), "ALERTMANAGER_PORT": "19093",
                   "MONITORING_ARTIFACT_ROOT": str(root), "MONITORING_UID": str(os.getuid()),
                   "MONITORING_GID": str(os.getgid()), "GRAFANA_ADMIN_PASSWORD": "ci-grafana-password",
                   "METRICS_PORT": "19108", "PROMETHEUS_PORT": "19090", "GRAFANA_PORT": "13000"}
            compose = ["docker", "compose", "-p", f"aethergate-monitor-{os.getpid()}", "-f", str(ROOT / "monitoring/compose.yaml"), "-f", str(ROOT / "monitoring/compose.test.yaml")]
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
            children = []
            def spawn_experiment(kind, name, pause=False):
                output = root / name
                probe = """
import os, runpy, signal, time
from research.evolution.checkpoints import Checkpoint
original = Checkpoint.save
def interrupt(self, state):
    original(self, state)
    progress = state.get('weight_state') if self.kind == 'formulas' else state
    if not state['completed'] and progress and progress['next_generation'] == 1:
        if PAUSE:
            time.sleep(300)
        else:
            os.kill(os.getpid(), signal.SIGKILL)
Checkpoint.save = interrupt
runpy.run_path('run_evolution.py', run_name='__main__')
""".replace("PAUSE", repr(pause))
                overrides = [f"experiment={kind}", f"hydra.run.dir={output}",
                             "experiment.parameters.agents=4", "experiment.parameters.steps=2"]
                if kind == "formulas":
                    overrides += ["experiment.parameters.architecture_population=2", "experiment.parameters.architecture_elites=1",
                                  "experiment.parameters.weight_population=2", "experiment.parameters.weight_elites=1",
                                  "experiment.parameters.weight_generations=2"]
                else:
                    overrides += ["experiment.parameters.runners=2", "experiment.parameters.blockers=2",
                                  "experiment.parameters.runner_elites=1", "experiment.parameters.blocker_elites=1",
                                  "experiment.parameters.generations=2"]
                log = (root / f"{name}.log").open("w")
                process = subprocess.Popen([sys.executable, "-c", probe, *overrides], cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                children.append((process, log))
                return process, output
            try:
                subprocess.run([*compose, "run", "--rm", "--no-deps", "--entrypoint", "/bin/promtool",
                    "-v", str(ROOT / "monitoring/alerts.test.yml") + ":/etc/prometheus/alerts.test.yml:ro",
                    "prometheus", "test", "rules", "/etc/prometheus/alerts.test.yml"],
                    cwd=ROOT, env=env, check=True, timeout=90)
                subprocess.run([*compose, "run", "--rm", "--no-deps", "--entrypoint", "/bin/amtool",
                    "alertmanager", "check-config", "/etc/alertmanager/alertmanager.yml"],
                    cwd=ROOT, env=env, check=True, timeout=90)
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
                self.assertEqual(len(dashboard["dashboard"]["panels"]), 11)
                for panel in dashboard["dashboard"]["panels"]:
                    for target in panel["targets"]:
                        expression = target["expr"].replace("$experiment", ".*").replace("$run", ".*")
                        if panel["id"] == 11:
                            wait(lambda: query(expression), "Grafana firing-alert panel")
                        else:
                            self.assertTrue(query(expression), f"Dashboard query has no data: {expression}")
                proxy = wait(lambda: get("http://127.0.0.1:13000/api/datasources/proxy/uid/aethergate-prometheus/api/v1/query?" +
                    urlencode({"query": "aethergate_population_reward"}), True), "Grafana datasource proxy")
                self.assertTrue(proxy["data"]["result"])
                def firing():
                    return [alert for alert in get("http://127.0.0.1:19090/api/v1/alerts")["data"]["alerts"] if alert["state"] == "firing"]
                def received(alert_name, run_name, state="firing"):
                    notifications = get("http://127.0.0.1:19109/notifications")
                    return any(alert.get("status") == state and alert["labels"].get("alertname") == alert_name
                               and alert["labels"].get("run") == run_name
                               for notification in notifications for alert in notification["alerts"])
                def expect_alert(alert_name, run_name):
                    wait(lambda: any(alert["labels"].get("alertname") == alert_name and alert["labels"].get("run") == run_name
                                     for alert in firing()), f"Prometheus firing {alert_name}/{run_name}")
                    wait(lambda: any(alert["labels"].get("alertname") == alert_name and alert["labels"].get("run") == run_name
                                     for alert in get("http://127.0.0.1:19093/api/v2/alerts")), f"Alertmanager reception {alert_name}/{run_name}")
                    wait(lambda: received(alert_name, run_name), f"webhook delivery {alert_name}/{run_name}")
                for kind in ("formulas", "selfplay"):
                    expect_alert("AetherGateNonFinite", f"{kind}-nonfinite")
                    expect_alert("AetherGateDiversityCollapse", f"{kind}-collapse")
                for module, kind, mutation in ((formulas, "formulas", "mutate_weight_elites"),
                                               (selfplay, "selfplay", "mutate_elites")):
                    original = getattr(module, mutation)
                    def corrupt(*args, **kwargs):
                        population = original(*args, **kwargs)
                        return population._replace(output_c=population.output_c.at[-1, 0, 0].set(float("nan")))
                    with patch.object(module, mutation, side_effect=corrupt):
                        snapshot = run_case(module, root / f"{kind}-nan", failure="nan")
                    self.assertEqual(snapshot.stop.code, "nonfinite")
                    expect_alert("AetherGateNonFinite", f"{kind}-nan")
                long_process, long_output = spawn_experiment("formulas", "long-running", pause=True)
                wait(lambda: (long_output / "checkpoint.npz").exists() and read_snapshot(long_output / "metrics.json").roles,
                     "long-running generation with independent heartbeat")
                for kind in ("formulas", "selfplay"):
                    process, output = spawn_experiment(kind, f"{kind}-killed")
                    self.assertEqual(process.wait(timeout=90), -signal.SIGKILL)
                    write_exit_marker(output, process.returncode)
                    expect_alert("AetherGateUnexpectedStop", output.name)
                process, output = spawn_experiment("selfplay", "silent-kill")
                self.assertEqual(process.wait(timeout=90), -signal.SIGKILL)
                self.assertFalse((output / "run_exit.json").exists())
                expect_alert("AetherGateHeartbeatLost", "silent-kill")
                self.assertIsNone(long_process.poll())
                self.assertGreater(float(query('time() - aethergate_run_updated_timestamp_seconds{run="long-running"}')[0]["value"][1]), 30)
                self.assertLess(float(query('time() - aethergate_run_heartbeat_timestamp_seconds{run="long-running"}')[0]["value"][1]), 10)
                protected = {"formulas-healthy", "selfplay-healthy", "long-running"}
                self.assertFalse(any(alert["labels"].get("run") in protected for alert in firing()))
                long_process.send_signal(signal.SIGINT)
                long_process.wait(timeout=30)
                wait(lambda: query('aethergate_run_status{run="long-running",status="interrupted"} == 1'), "intentional interruption status")
                self.assertFalse(any(alert["labels"].get("run") == "long-running" for alert in firing()))
                run_case(formulas, root / "formulas-nonfinite")
                wait(lambda: not any(alert["labels"].get("run") == "formulas-nonfinite" for alert in firing()), "Prometheus alert resolution")
                wait(lambda: received("AetherGateNonFinite", "formulas-nonfinite", "resolved"), "resolved webhook delivery")
                print("PASS alerts: real NaN/Inf, diversity collapse, SIGKILL and missing heartbeat fired in Prometheus, reached Alertmanager and webhook; healthy/long-running/intentional stops stayed clear; recovery resolved the alert", flush=True)
                print("PASS monitoring: real metric scrape, live update, Grafana dashboard and datasource queries succeeded", flush=True)
            finally:
                for process, log in children:
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    log.close()
                subprocess.run([*compose, "logs", "--no-color", "--tail=30"], env=env, check=False, timeout=30)
                subprocess.run([*compose, "down", "--volumes", "--remove-orphans"], env=env, check=False, timeout=60)


if __name__ == "__main__":
    unittest.main()
