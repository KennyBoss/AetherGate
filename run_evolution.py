#!/usr/bin/env python3
"""Compose Hydra experiment settings before importing JAX or the engine."""

import runpy
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "hypothesis": "run_hypothesis_loop.py",
    "formulas": "evolve_ssm_formulas.py",
    "selfplay": "coevolve_ssm_selfplay.py",
}


def experiment_argv(config: DictConfig) -> tuple[Path, list[str]]:
    experiment = OmegaConf.to_container(config.experiment, resolve=True)
    if not isinstance(experiment, dict) or set(experiment) != {"kind", "parameters"}:
        raise ValueError("experiment must contain kind and parameters")
    kind = experiment["kind"]
    if not isinstance(kind, str) or kind not in SCRIPTS:
        raise ValueError(f"Unknown experiment kind: {kind!r}")
    parameters = experiment["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a mapping")
    argv = []
    for key, value in parameters.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"Invalid parameter name: {key!r}")
        if value is None:
            continue
        if type(value) not in (str, int, float):
            raise ValueError(f"Parameter {key} must be a string or number")
        argv.append(f"--{key.replace('_', '-')}={value}")
    return ROOT / "research" / "evolution" / SCRIPTS[kind], argv


@hydra.main(version_base="1.3", config_path="configs/evolution", config_name="config")
def main(config: DictConfig) -> None:
    script, argv = experiment_argv(config)
    previous_argv = sys.argv
    previous_path = sys.path[:]
    try:
        sys.argv = [str(script), *argv]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
