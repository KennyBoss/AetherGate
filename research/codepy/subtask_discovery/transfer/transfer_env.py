"""Read-only accepted reference and existing search/runtime modules."""
import os,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'research/codepy/subtask_discovery/local_completion'))
import context as frozen
from common import legacy,previous
from register_abi import tables,execute_local,numpy_pair
REF=frozen.OUT
OUT=ROOT/'artifacts/codepy_transfer_20261101'
ALIAS=ROOT/'artifacts/codepy_references/codepy_local_completion.json'
prior=frozen.prior
save,sha,vm=prior.save,prior.sha,prior.vm
