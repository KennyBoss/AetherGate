"""Failure injection must stop real loops before ranking or another mutation."""

from collections import namedtuple
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.evolution.guardrails import GuardrailStop, check_population
from research.evolution import evolve_ssm_formulas as formulas
from research.evolution import coevolve_ssm_selfplay as selfplay


class PopulationHealthTests(unittest.TestCase):
    def test_all_weights_contribute_and_singleton_is_not_collapse(self):
        Genome = namedtuple("Genome", "weights bias")
        healthy = Genome(np.array([[0., 1.], [1., 0.]]), np.zeros((2, 1)))
        self.assertGreater(check_population(healthy, 1e-6, stage="test"), 0)
        self.assertIsNone(check_population(Genome(np.ones((1, 2)), np.zeros((1, 1))), 1e-6, stage="test"))
        with self.assertRaises(GuardrailStop) as stopped:
            check_population(Genome(np.ones((2, 2)), np.zeros((2, 1))), 1e-6, stage="test")
        self.assertEqual(stopped.exception.reason["code"], "diversity_collapse")
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(GuardrailStop) as stopped:
                check_population(Genome(np.array([[value, 1.]]), np.zeros((1, 1))), 1e-6, stage="test")
            self.assertEqual(stopped.exception.reason["code"], "nonfinite")


class LoopStopTests(unittest.TestCase):
    def args(self, module, output):
        flags = ["test", "--agents=4", "--steps=2", f"--output-json={output}"]
        if module is formulas:
            flags += ["--architecture-population=2", "--architecture-elites=1",
                      "--weight-population=2", "--weight-elites=1", "--weight-generations=3"]
        else:
            flags += ["--runners=2", "--blockers=2", "--runner-elites=1", "--blocker-elites=1", "--generations=3"]
        with patch.object(sys, "argv", flags):
            return module.parse_args()

    def stopped(self, module, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as exit:
            module.run(args)
        self.assertEqual(exit.exception.code, 2)
        raw = Path(args.output_json).read_text()
        payload = json.loads(raw, parse_constant=lambda value: self.fail(f"Invalid JSON number: {value}"))
        self.assertEqual(payload["status"], "stopped")
        self.assertEqual(payload["seed"], args.seed)
        return payload["stop_reason"]

    def test_mutation_nan_and_infinities_stop_before_next_evaluation(self):
        for module, mutate_name, evaluate_name in (
            (formulas, "mutate_weight_elites", "evaluate_population"),
            (selfplay, "mutate_elites", "evaluate_populations"),
        ):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(module=module.__name__, value=value), tempfile.TemporaryDirectory() as directory:
                    args = self.args(module, Path(directory) / "result.json")
                    original = getattr(module, mutate_name)
                    def corrupt(*a, **kw):
                        genome = original(*a, **kw)
                        return genome._replace(output_c=genome.output_c.at[-1, 0, 0].set(value))
                    with patch.object(module, mutate_name, side_effect=corrupt) as mutate, \
                         patch.object(module, evaluate_name, wraps=getattr(module, evaluate_name)) as evaluate:
                        reason = self.stopped(module, args)
                    self.assertEqual(mutate.call_count, 1)
                    self.assertEqual(evaluate.call_count, 2 if module is formulas else 1)  # Formula warmup + gen 0.
                    self.assertEqual(reason["code"], "nonfinite")
                    self.assertEqual(reason["field"], "output_c")
                    self.assertEqual(reason["generation"], 1)
                    self.assertEqual(reason["stage"], "after_mutation")

    def test_nonfinite_metric_never_reaches_selection(self):
        for module, evaluate_name, field in (
            (formulas, "evaluate_population", "policy_entropy"),
            (selfplay, "evaluate_populations", "payoff_matrix"),
        ):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(module=module.__name__, value=value), tempfile.TemporaryDirectory() as directory:
                    args = self.args(module, Path(directory) / "result.json")
                    original = getattr(module, evaluate_name)
                    calls = 0
                    def corrupt(*a, **kw):
                        nonlocal calls
                        calls += 1
                        metrics = original(*a, **kw)
                        if module is formulas and calls == 1 and not np.isnan(value):
                            return metrics
                        return metrics._replace(**{field: getattr(metrics, field).at[0].set(value)})
                    with patch.object(module, evaluate_name, side_effect=corrupt), \
                         patch.object(module.jnp, "argsort", side_effect=AssertionError("Unsafe selection")):
                        reason = self.stopped(module, args)
                    self.assertEqual(reason["code"], "nonfinite")
                    self.assertEqual(reason["field"], field)
                    self.assertEqual(reason["stage"], "warmup" if module is formulas and np.isnan(value) else "evaluation")

    def test_blocker_collapse_is_checked_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(selfplay, Path(directory) / "result.json")
            original = selfplay.mutate_elites
            def collapse_blockers(key, elites, size, sigma):
                # Let runners remain healthy and collapse only the second role.
                return original(key, elites, size, sigma if mutate.call_count == 1 else 0.)
            with patch.object(selfplay, "mutate_elites", side_effect=collapse_blockers) as mutate:
                reason = self.stopped(selfplay, args)
            self.assertEqual(reason["code"], "diversity_collapse")
            self.assertEqual(reason["population"], "blockers")
            self.assertEqual(reason["generation"], 1)

    def test_invalid_threshold_and_nonfinite_configuration_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(formulas, Path(directory) / "result.json")
            for threshold in (0., -1., np.nan, np.inf):
                args.min_population_diversity = threshold
                reason = self.stopped(formulas, args)
                self.assertEqual(reason["stage"], "configuration")

    def test_direct_run_without_output_path_saves_fallback_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(selfplay, None)
            args.output_json = None
            args.min_population_diversity = -1.
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    selfplay.run(args)
                payload = json.loads(Path("guardrail_stop.json").read_text())
            finally:
                os.chdir(previous)
            self.assertEqual(payload["stop_reason"]["code"], "invalid_configuration")


if __name__ == "__main__":
    unittest.main()
