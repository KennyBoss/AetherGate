import unittest

from run_kv_scale_law import (
    CAPACITY_POINTS,
    TASK_POINTS,
    consecutive,
    gate_status,
    kv_parameter_count,
    phase_status,
    runtime_state_bytes,
    transformer_cache_bytes,
)


class KVScaleLawTest(unittest.TestCase):
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

    def test_two_adjacent_negatives_fail(self) -> None:
        results = [
            {"point": {"name": TASK_POINTS[0].name}, "status": "NEGATIVE", "pareto_positive": False},
            {"point": {"name": TASK_POINTS[1].name}, "status": "NEGATIVE", "pareto_positive": False},
        ]
        self.assertTrue(consecutive([True, True], 2))
        self.assertEqual(gate_status(results, lm_complete=False, mamba_complete=False)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
