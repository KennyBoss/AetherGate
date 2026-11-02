"""Frozen prior implementation and results are read-only dependencies."""
import os,sys
from pathlib import Path
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
ROOT=Path(__file__).resolve().parents[4]
GUIDED=ROOT/'research/codepy/subtask_discovery/branch_guided'
sys.path.insert(0,str(GUIDED))
import run_guided as prior
from common import legacy,previous
BASE=ROOT/'artifacts/codepy_branch_guided_20261101'
OUT=ROOT/'artifacts/codepy_local_completion_20261101'
