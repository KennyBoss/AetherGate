"""Atomic, pickle-free snapshots of experiment state and random generators."""

import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import zipfile

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def add_checkpoint_arguments(parser):
    parser.add_argument("--checkpoint-path", default=None, help="Atomic snapshot destination; defaults beside output JSON.")
    parser.add_argument("--resume-from", default=None, help="Resume an identical experiment from an existing checkpoint.")


def restore_array(value, template):
    array = np.asarray(value)
    if array.shape != template.shape or array.dtype != template.dtype or not np.isfinite(array).all():
        raise ValueError("Invalid checkpoint array shape, dtype or nonfinite contents")
    return jnp.asarray(array)


def restore_genome(values, template):
    if set(values) != set(template._fields):
        raise ValueError("Invalid checkpoint genome fields")
    return type(template)(*(restore_array(values[name], leaf) for name, leaf in zip(template._fields, template)))


class Checkpoint:
    def __init__(self, kind, args):
        self.kind = kind
        self.path = Path(args.checkpoint_path or (
            Path(args.output_json).with_suffix(".checkpoint.npz") if args.output_json else f"{kind}.checkpoint.npz"))
        self.source = args.resume_from
        code = hashlib.sha256()
        for name in ("agent_ssm_core.py", "requirements.lock", "research/evolution/checkpoints.py",
                     "research/evolution/guardrails.py", "research/evolution/reproducibility.py",
                     "research/evolution/evolve_ssm_formulas.py", "research/evolution/coevolve_ssm_selfplay.py"):
            code.update(name.encode())
            code.update((ROOT / name).read_bytes())
        self.header = {
            "version": 1, "kind": kind,
            "parameters": {key: value for key, value in vars(args).items()
                           if key not in {"output_json", "checkpoint_path", "resume_from"}},
            "environment": {"python": platform.python_version(), "jax": jax.__version__,
                            "jaxlib": jaxlib.__version__, "numpy": np.__version__,
                            "backend": jax.default_backend(), "code_sha256": code.hexdigest()},
        }

    def save(self, state):
        arrays = {}
        def encode(value):
            if isinstance(value, (jax.Array, np.ndarray)):
                array = np.asarray(value)
                if array.dtype.hasobject or not np.isfinite(array).all():
                    raise ValueError("Cannot checkpoint nonfinite or object arrays")
                name = f"array_{len(arrays)}"
                arrays[name] = array
                return {"__ndarray__": name}
            if isinstance(value, dict):
                return {key: encode(child) for key, child in value.items()}
            if isinstance(value, (tuple, list)):
                return [encode(child) for child in value]
            return value
        metadata = json.dumps({**self.header, "state": encode(state)}, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=f".{self.path.name}.", delete=False) as stream:
                temporary = stream.name
                np.savez(stream, metadata=np.asarray(metadata), **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    def load(self):
        if not self.source:
            return None
        try:
            with np.load(self.source, allow_pickle=False) as archive:
                def reject_constant(value):
                    raise ValueError(f"Invalid number {value}")
                metadata = json.loads(archive["metadata"].item(), parse_constant=reject_constant)
                for key, expected in self.header.items():
                    if metadata[key] != expected:
                        raise ValueError(f"Checkpoint {key} does not match this experiment")
                def decode(value):
                    if isinstance(value, dict):
                        if set(value) == {"__ndarray__"}:
                            array = archive[value["__ndarray__"]]
                            if array.dtype.hasobject or not np.isfinite(array).all():
                                raise ValueError("Invalid checkpoint array")
                            return array
                        return {key: decode(child) for key, child in value.items()}
                    if isinstance(value, list):
                        return [decode(child) for child in value]
                    return value
                state = decode(metadata["state"])
                if not isinstance(state, dict):
                    raise ValueError("Checkpoint state must be a mapping")
                return state
        except (ValueError, TypeError, KeyError, OSError, EOFError, zipfile.BadZipFile) as error:
            raise ValueError(f"Cannot resume checkpoint: {error}") from error
