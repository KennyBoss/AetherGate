"""Shared program representation and frozen block loading; no task oracles."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from research.codepy.ssm_dispatch.catalog import ROOT,digest,release_path

class CertificationError(ValueError):
    pass

def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def input_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def blocks_for(catalog):
    path=ROOT/f'artifacts/codepy_transfer_20261101/frozen_library_seed{catalog.seed}.json'
    protocol=json.loads((path.parent/'protocol.json').read_text())
    if digest(path)!={str(release_path(k)): v for k, v in protocol['library_manifest'].items()}[str(path)]:raise ValueError('Frozen source block hash mismatch')
    library=json.loads(path.read_text())
    for name,h in library['provenance'].items():
        if digest(name)!=h:raise ValueError('Block provenance hash mismatch')
    return library['blocks']

@dataclass(frozen=True)
class Candidate:
    seed: int
    task: str
    program: tuple[int,...]
    arity: int = 4
    def payload(self):
        value={'seed':self.seed,'task':self.task,'program':list(self.program)}
        if self.arity!=4:value['arity']=self.arity
        return value

@dataclass(frozen=True)
class DevelopmentResult:
    status: str
    command: int | None
    evidence: dict

def verifier_for(catalog,entry):
    mode=entry.get('mode')
    if mode=='hierarchical-v1':
        from research.codepy.ssm_dispatch.hierarchy_v1 import Verifier
        return Verifier(catalog,entry['language'],entry.get('certificate'))
    if mode=='hierarchical-v2':
        from research.codepy.ssm_dispatch.hierarchy import Verifier
        return Verifier(catalog,entry['language'],entry.get('certificate'))
    if mode=='examples-v1':
        from research.codepy.ssm_dispatch.examples_v1 import Verifier
    elif mode=='examples-v2':
        from research.codepy.ssm_dispatch.examples import Verifier
    elif mode is None:
        from research.codepy.ssm_dispatch.development import Verifier
    else:raise CertificationError('Unknown program verification mode')
    return Verifier(catalog)


def vector_arguments(value):
    x=np.asarray(value)
    if x.dtype!=np.float32 or x.ndim!=2 or x.shape[1]<2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Arguments must be nonempty finite float32 (N, arity), arity>=2')
    return x
