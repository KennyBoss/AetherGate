"""Certified local-completion programs behind a validated numeric interface."""
from __future__ import annotations
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[3]
LIBRARY = ROOT / 'artifacts/codepy_local_completion_20261101'
TASKS = ('sort4', 'reverse4', 'rotate4', 'descending4')
PERMS = np.asarray(list(itertools.permutations(range(4))), np.int32)

def release_path(path):
    """Resolve historical provenance paths inside this checkout, never another tree."""
    path = Path(path)
    if not path.is_absolute():
        return ROOT / path
    try:
        path.relative_to(ROOT)
        return path
    except ValueError:
        for anchor in ('artifacts', 'research'):
            if anchor in path.parts:
                return ROOT.joinpath(*path.parts[path.parts.index(anchor):])
        raise ValueError('Unsupported external release dependency: ' + str(path))


def digest(path):
    return hashlib.sha256(release_path(path).read_bytes()).hexdigest()

def arguments(value):
    x = np.asarray(value)
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Arguments must be a nonempty finite float32 array of shape (N, 4)')
    return x

class CertifiedCatalog:
    def __init__(self, seed: int):
        if type(seed) is not int or seed not in range(5):
            raise ValueError('Seed must be 0..4')
        reference = json.loads((ROOT / 'artifacts/codepy_references/codepy_local_completion.json').read_text())
        if digest(LIBRARY / 'results.json') != reference['results_sha256']:
            raise ValueError('Certified reference hash mismatch')
        result = json.loads((LIBRARY / 'results.json').read_text())
        for name, expected in result['protocol']['source_sha256'].items():
            if digest(name) != expected:
                raise ValueError('Certified executor source hash mismatch: ' + name)
        if result.get('wins') != 20 or result.get('candidate_status') != 'accepted':
            raise ValueError('An accepted 20/20 library is required')
        for name, expected in json.loads((LIBRARY / 'library_manifest.json').read_text()).items():
            if digest(name) != expected:
                raise ValueError('Library hash mismatch: ' + name)
        self.seed = seed
        self.version = digest(LIBRARY / 'results.json')
        sys.path.insert(0, str(ROOT / 'research/codepy/subtask_discovery/branch_guided'))
        import run_guided as g
        self.g = g
        self.selected = [next(r for r in result['runs'] if r['seed'] == seed and r['task'] == t) for t in TASKS]
        self.executors = []
        self.references = []
        old = json.loads((g.previous.OLD / f'library_seed{seed}.json').read_text())
        self.entries = [[v] for v in g.s.ALPHABET] + old['blocks']
        self.table, self.lengths = g.s.table_for(old['blocks'])
        self.contracts = g.baseline.calls.c.contract_tables(self.entries)
        # Argument-ABI champions depend on this frozen primitive/block library too.
        self.argument_library_sha256 = digest(g.previous.OLD / f'library_seed{seed}.json')
        for row in self.selected:
            q = row['branches'][row['chosen_by_TRAIN']]
            if not q['certificate']['passed'] or not q['certificate']['independent_agrees']:
                raise ValueError('Uncertified program')
            if q['search_kind'] == 'arguments':
                program = jnp.asarray([q['program']])
                def execute(x, program=program):
                    return g.baseline.calls.execute(program, x, self.table, self.lengths, self.contracts, n=4, primitive_count=len(g.s.ALPHABET))[0]
                def reference(x, q=q):
                    return g.baseline.calls.numpy_execute(q['program'], x, self.entries, len(g.s.ALPHABET))
            else:
                codes, layouts, returns = g.tables(q['blocks'])
                program = jnp.asarray([q['program']])
                def execute(x, program=program, codes=codes, layouts=layouts, returns=returns):
                    return g.compose(program, x, codes, layouts, returns)[0]
                def reference(x, q=q):
                    return g.numpy_compose(q['program'], x, q['blocks'])
            self.executors.append(jax.jit(execute))
            self.references.append(reference)

    def execute(self, command: int, values, *, independent=False):
        if type(command) is not int or not 0 <= command < 96:
            raise ValueError('Unknown command ID')
        x = arguments(values)
        task, pi = divmod(command, 24)
        p = PERMS[pi]
        executor = self.references[task] if independent else self.executors[task]
        result = np.asarray(executor(x[:, p]))[:, np.argsort(p)]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite execution result')
        return result

    def candidates(self, values):
        return np.stack([self.execute(i, values) for i in range(96)])

@lru_cache(maxsize=5)
def catalog(seed):
    return CertifiedCatalog(seed)
