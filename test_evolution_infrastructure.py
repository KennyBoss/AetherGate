"""Contracts and real CPU smoke runs through the configured entry point."""

import json
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pydantic import ValidationError

from research.evolution.hypothesis_contract import FormulaModel, HypothesisModel
from run_evolution import ROOT, experiment_argv


FORMULA = dict(memory_op="identity", input_op="identity", combine_op="add", final_op="tanh")


class ContractTests(unittest.TestCase):
    def test_nested_and_legacy_formats(self):
        for proposal in ({"formula": FORMULA}, FORMULA, {"id": "legacy", **FORMULA}):
            self.assertEqual(HypothesisModel.model_validate(proposal).formula.model_dump(), FORMULA)
        self.assertEqual(FORMULA["memory_op"], "identity")

    def test_bad_types_fields_and_operations(self):
        for proposal in (
            None, [], {"formula": []}, {"formula": {"memory_op": "identity"}},
            {"formula": {**FORMULA, "memory_op": "__import__('os')"}},
            {"formula": {**FORMULA, "input_op": ["identity"]}},
            {"formula": {**FORMULA, "code": "print(1)"}},
            {"formula": FORMULA, "id": 7}, {"formula": FORMULA, "unknown": True},
        ):
            with self.subTest(proposal=proposal), self.assertRaises(ValidationError):
                HypothesisModel.model_validate(proposal)

    def test_schema_is_formal_and_closed(self):
        schema = HypothesisModel.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertFalse(schema["$defs"]["FormulaModel"]["additionalProperties"])
        self.assertEqual(set(schema["$defs"]["FormulaModel"]["required"]), set(FORMULA))
        self.assertIn("tanh", FormulaModel.model_json_schema()["properties"]["final_op"]["enum"])

    def test_file_and_generator_use_same_parser(self):
        with patch.object(sys, "path", [str(ROOT / "research/evolution"), *sys.path]):
            loop = importlib.import_module("run_hypothesis_loop")
        proposal = {"formula": FORMULA}
        for text in (json.dumps([proposal]), "# proposals\n" + json.dumps(proposal)):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.json"
                path.write_text(text)
                self.assertEqual(loop.load_json_or_jsonl(str(path)), [proposal])
                self.assertEqual(loop.parse_proposals_text(text), [proposal])
        for text in ('[null]', '[1, {}]', 'null', '{broken'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loop.parse_proposals_text(text)
        self.assertIsNone(loop.validate_proposal({"formula": FORMULA, "extra": 1}, 0)[0])


class ConfigurationTests(unittest.TestCase):
    def test_composition_override_and_invalid_launch(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs/evolution")):
            for kind in ("hypothesis", "formulas", "selfplay"):
                config = compose(config_name="config", overrides=[f"experiment={kind}",
                    "experiment.parameters.seed=17", "experiment.parameters.output_json=null"])
                if kind == "hypothesis":
                    config.experiment.parameters.feedback_jsonl = None
                    config.experiment.parameters.prompt_json = None
                if kind != "hypothesis":
                    config.experiment.parameters.checkpoint_path = None
                script, argv = experiment_argv(config)
                self.assertTrue(script.exists())
                self.assertIn("--seed=17", argv)
        for experiment in ({"kind": "../evil", "parameters": {}},
                           {"kind": "selfplay", "parameters": {"agents": [1]}},
                           {"kind": "selfplay", "parameters": {"--help": 1}}):
            with self.assertRaises(ValueError):
                experiment_argv(OmegaConf.create({"experiment": experiment}))

    def test_real_configured_runs(self):
        env = {**os.environ, "JAX_PLATFORM_NAME": "cpu", "JAX_PLATFORMS": "cpu"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposals = root / "proposals.json"
            proposals.write_text(json.dumps([
                {"id": "valid", "formula": FORMULA},
                {"id": "bad", "formula": {**FORMULA, "input_op": "exec"}},
                {"id": "duplicate", "formula": FORMULA},
            ]))
            for kind in ("hypothesis", "formulas", "selfplay"):
                out = root / kind
                overrides = [f"experiment={kind}", f"hydra.run.dir={out}",
                             "experiment.parameters.agents=4", "experiment.parameters.steps=2"]
                if kind == "selfplay":
                    overrides += [f"experiment.parameters.{key}=2" for key in ("runners", "blockers")]
                    overrides += ["experiment.parameters.generations=1"]
                else:
                    overrides += ["experiment.parameters.weight_population=2",
                                  "experiment.parameters.weight_generations=1"]
                    if kind == "formulas":
                        overrides += ["experiment.parameters.weight_elites=1",
                                      "experiment.parameters.architecture_population=2",
                                      "experiment.parameters.architecture_elites=1"]
                    else:
                        overrides += [f"experiment.parameters.proposals={proposals}"]
                proc = subprocess.run([sys.executable, str(ROOT / "run_evolution.py"), *overrides],
                                      cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertTrue((out / ".hydra/config.yaml").exists())
                result = json.loads((out / "result.json").read_text())
                self.assertEqual(result["backend"], "cpu")
                if kind == "hypothesis":
                    self.assertEqual(result["protocol"]["accepted"], 1)
                    self.assertEqual(result["protocol"]["rejected"], 2)
                    prompt = json.loads((out / "prompt.json").read_text())
                    self.assertIn("$defs", prompt["schema"])
                    self.assertEqual(len((out / "feedback.jsonl").read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
