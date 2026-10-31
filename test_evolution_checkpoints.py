"""Checkpoint replay, safe persistence, compatibility and interrupted boundaries."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.evolution.checkpoints import Checkpoint, restore_array
from research.evolution import evolve_ssm_formulas as formulas
from research.evolution import coevolve_ssm_selfplay as selfplay
from research.evolution.reproducibility import canonical_result


class CheckpointTests(unittest.TestCase):
    def args(self, module, directory):
        directory = Path(directory)
        flags = ["test", "--agents=4", "--steps=2", f"--output-json={directory / 'result.json'}",
                 f"--checkpoint-path={directory / 'checkpoint.npz'}"]
        if module is formulas:
            flags += ["--architecture-population=2", "--architecture-elites=1", "--architecture-generations=1",
                      "--weight-population=2", "--weight-elites=1", "--weight-generations=1"]
        else:
            flags += ["--runners=2", "--blockers=2", "--runner-elites=1", "--blocker-elites=1", "--generations=3"]
        with patch.object(sys, "argv", flags):
            return module.parse_args()

    def run_result(self, module, args):
        with contextlib.redirect_stdout(io.StringIO()):
            module.run(args)
        return canonical_result(json.loads(Path(args.output_json).read_text()))

    def test_resumed_results_match_at_nested_and_late_boundaries(self):
        for module in (formulas, selfplay):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as directory:
                full = self.args(module, Path(directory) / "full")
                expected = self.run_result(module, full)
                for late in (False, True):
                    partial = self.args(module, Path(directory) / f"partial-{late}")
                    original = Checkpoint.save
                    def interrupt(session, state):
                        original(session, state)
                        if module is formulas:
                            progress = state.get("weight_state")
                            ready = progress is not None and progress["next_generation"] == 1
                            if late:
                                ready = ready and state["architecture_generation"] == 1 and state["architecture_index"] == 1
                        else:
                            ready = state["next_generation"] == (3 if late else 1)
                        if ready:
                            raise SystemExit(143)
                    with patch.object(Checkpoint, "save", interrupt), contextlib.redirect_stdout(io.StringIO()), \
                         self.assertRaises(SystemExit) as stopped:
                        module.run(partial)
                    self.assertEqual(stopped.exception.code, 143)
                    self.assertFalse(Path(partial.output_json).exists())
                    resumed = self.args(module, Path(directory) / f"resumed-{late}")
                    resumed.resume_from = partial.checkpoint_path
                    self.assertEqual(expected, self.run_result(module, resumed))
                    # Completion can also be recovered without evaluating again.
                    complete = self.args(module, Path(directory) / f"complete-{late}")
                    complete.resume_from = resumed.checkpoint_path
                    evaluation = "evaluate_population" if module is formulas else "evaluate_populations"
                    with patch.object(module, evaluation, side_effect=AssertionError("Completed run was recomputed")):
                        self.assertEqual(expected, self.run_result(module, complete))

    def test_failed_atomic_write_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(selfplay, directory)
            session = Checkpoint("selfplay", args)
            state = {"key": np.array([1, 2], dtype=np.uint32), "generation": 1}
            session.save(state)
            before = Path(args.checkpoint_path).read_bytes()
            with patch("research.evolution.checkpoints.np.savez", side_effect=OSError("disk full")), self.assertRaises(OSError):
                session.save({"generation": 2})
            self.assertEqual(before, Path(args.checkpoint_path).read_bytes())
            self.assertEqual(list(Path(directory).iterdir()), [Path(args.checkpoint_path)])
            args.resume_from = args.checkpoint_path
            restored = Checkpoint("selfplay", args).load()
            np.testing.assert_array_equal(restored["key"], state["key"])

    def test_wrong_seed_kind_environment_and_corrupt_archive_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(selfplay, directory)
            session = Checkpoint("selfplay", args)
            session.save({"completed": False})
            args.resume_from = args.checkpoint_path
            args.seed += 1
            with self.assertRaisesRegex(ValueError, "parameters"):
                Checkpoint("selfplay", args).load()
            args.seed -= 1
            with self.assertRaisesRegex(ValueError, "kind"):
                Checkpoint("formulas", args).load()
            reader = Checkpoint("selfplay", args)
            reader.header["environment"]["jax"] = "other"
            with self.assertRaisesRegex(ValueError, "environment"):
                reader.load()
            Path(args.checkpoint_path).write_bytes(b"truncated checkpoint")
            with self.assertRaisesRegex(ValueError, "Cannot resume checkpoint"):
                Checkpoint("selfplay", args).load()

    def test_arrays_are_not_cast_or_unpickled(self):
        template = np.zeros((2, 3), dtype=np.float32)
        for value in (np.zeros((3, 2), dtype=np.float32), np.zeros((2, 3), dtype=np.float64),
                      np.full((2, 3), np.nan, dtype=np.float32)):
            with self.assertRaises(ValueError):
                restore_array(value, template)
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(selfplay, directory)
            session = Checkpoint("selfplay", args)
            with self.assertRaises(ValueError):
                session.save({"population": np.array([object()])})
            np.savez(args.checkpoint_path, metadata=np.asarray({"unsafe": True}, dtype=object))
            args.resume_from = args.checkpoint_path
            with self.assertRaisesRegex(ValueError, "Cannot resume checkpoint"):
                Checkpoint("selfplay", args).load()


if __name__ == "__main__":
    unittest.main()
