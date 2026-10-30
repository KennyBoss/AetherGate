"""Stable weight fingerprints and canonical computational results."""

import hashlib
import json

import numpy as np


TIMING_FIELDS = frozenset({
    "compile_s", "weight_evolution_s", "elapsed_s",
    "throughput_agent_steps_s", "throughput_pair_agent_steps_s",
})


def genome_sha256(genome):
    digest = hashlib.sha256()
    for name, leaf in zip(genome._fields, genome):
        array = np.asarray(leaf)
        metadata = json.dumps([name, array.dtype.str, array.shape], separators=(",", ":"))
        digest.update(metadata.encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_result(value):
    """Exclude only measured timings; retain every other current or future field."""
    def computational(item):
        if isinstance(item, dict):
            return {key: computational(child) for key, child in item.items() if key not in TIMING_FIELDS}
        if isinstance(item, list):
            return [computational(child) for child in item]
        return item

    return json.dumps(computational(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
