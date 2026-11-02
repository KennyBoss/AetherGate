import os
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

from jax_backend import configure_environment

from run_kv_scale_law import (
    CAPACITY_POINTS,
    TASK_POINTS,
    consecutive,
    gate_status,
    kv_parameter_count,
    lm_protocol_complete,
    phase_status,
    runtime_state_bytes,
    training_compute_units,
    transformer_cache_bytes,
    validate_mamba_reports,
)
from run_mamba_mqar import array_sha256, make_mqar_recall


class KVScaleLawTest(unittest.TestCase):
    def test_metal_backend_uses_jax_auto_discovery(self) -> None:
        with patch.dict("os.environ", {"JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"}):
            configure_environment("metal")
            self.assertNotIn("JAX_PLATFORM_NAME", os.environ)
            self.assertNotIn("JAX_PLATFORMS", os.environ)
            configure_environment("cpu")
            self.assertEqual(os.environ["JAX_PLATFORM_NAME"], "cpu")

    def test_registered_counts_and_order(self) -> None:
        self.assertEqual([point.name for point in TASK_POINTS], ["task-0", "task-1", "task-2", "task-3"])
        self.assertEqual(kv_parameter_count(TASK_POINTS[0]), 9661)
        for point in CAPACITY_POINTS:
            ratio = kv_parameter_count(point) / point.target_params
            self.assertGreaterEqual(ratio, 0.8)
            self.assertLessEqual(ratio, 1.25)

    def test_resource_estimates(self) -> None:
        point = TASK_POINTS[0]
        self.assertEqual(runtime_state_bytes(point), 4 * (96 + 64 + 64**2))
        self.assertEqual(transformer_cache_bytes(layers=1, context=67, d_model=32), 17_152)
        self.assertGreater(training_compute_units(CAPACITY_POINTS[1]), 10 * training_compute_units(CAPACITY_POINTS[0]))

    def test_lm_protocol_rejects_smoke_scale(self) -> None:
        run = {"budget_match": True, "train_tokens": 2_000_000, "eval_tokens": 2_000_000}
        smoke = Namespace(lm_streams=4, lm_seq_len=16, lm_tokens_per_stream=96)
        million = Namespace(lm_streams=128, lm_seq_len=128, lm_tokens_per_stream=8192)
        self.assertFalse(lm_protocol_complete(smoke, [run, run, run]))
        self.assertTrue(lm_protocol_complete(million, [run, run, run]))

    def test_gate_needs_both_phases_and_baselines(self) -> None:
        task_results = [
            {"point": {"name": point.name}, "status": "POSITIVE", "pareto_positive": True}
            for point in TASK_POINTS[:3]
        ]
        self.assertEqual(
            gate_status(task_results, lm_complete=False, mamba_complete=False)["status"],
            "INCOMPLETE",
        )
        all_results = task_results + [
            {"point": {"name": point.name}, "status": "POSITIVE", "pareto_positive": True}
            for point in CAPACITY_POINTS[:3]
        ]
        self.assertEqual(gate_status(all_results, lm_complete=True, mamba_complete=True)["status"], "PASS")
        self.assertEqual(phase_status(all_results, TASK_POINTS)["status"], "PASS")
        self.assertEqual(phase_status(all_results, CAPACITY_POINTS)["status"], "PASS")

    def test_mamba_report_validation_is_fail_closed(self) -> None:
        point = CAPACITY_POINTS[1]
        seeds = [11, 17, 23]
        runs = []
        for seed in seeds:
            train = make_mqar_recall(
                records=8,
                delay=point.delay,
                key_count=point.vocabulary,
                binding_count=point.bindings,
                query_count=point.queries,
                seed=seed,
            )
            evaluate = make_mqar_recall(
                records=4,
                delay=point.delay,
                key_count=point.vocabulary,
                binding_count=point.bindings,
                query_count=point.queries,
                seed=seed + 1,
            )
            runs.append({
                "seed": seed,
                "parameter_ratio": 1.0,
                "query_count": 4 * point.queries,
                "recall_accuracy": 0.5,
                "recall_loss": 1.0,
                "train_s": 1.0,
                "eval_s": 1.0,
                "train_data_sha256": array_sha256(*train),
                "eval_data_sha256": array_sha256(*evaluate),
                "torch_version": "test",
                "mamba_ssm_version": "test",
                "cuda_version": "test",
                "gpu": "test",
            })
        payload = {
            "kind": "kv_scale_law_mamba_mqar",
            "status": "COMPLETE",
            "implementation": {
                "official_repository": "https://github.com/state-spaces/mamba",
                "class": "mamba_ssm.Mamba",
                "block": "Mamba-1",
                "layers": 1,
            },
            "protocol": {
                "point": point.name,
                "seeds": seeds,
                "train_records": 8,
                "eval_records": 4,
                "key_count": point.vocabulary,
                "binding_count": point.bindings,
                "query_count": point.queries,
                "delay": point.delay,
                "seq_len": point.seq_len,
                "epochs": 2,
                "batch_size": point.batch_size,
                "learning_rate": 0.006,
                "grad_clip": 1.0,
                "query_only_loss": True,
            },
            "runs": runs,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "mamba.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            args = Namespace(
                mamba_report=[path], seeds="11,17,23", train_records=8, eval_records=4,
                epochs=2, learning_rate=0.006,
            )
            self.assertEqual(validate_mamba_reports(args, [point])["status"], "COMPLETE")
            payload["runs"][0]["train_data_sha256"] = "bad"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            self.assertEqual(validate_mamba_reports(args, [point])["status"], "INVALID")

    def test_two_adjacent_negatives_fail(self) -> None:
        results = [
            {"point": {"name": TASK_POINTS[0].name}, "status": "NEGATIVE", "pareto_positive": False},
            {"point": {"name": TASK_POINTS[1].name}, "status": "NEGATIVE", "pareto_positive": False},
        ]
        self.assertTrue(consecutive([True, True], 2))
        self.assertEqual(gate_status(results, lm_complete=False, mamba_complete=False)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
