import os,sys
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_PLATFORM_NAME']='cpu'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OLD_DIR=ROOT/'research/codepy/subtask_discovery/behavioral'
sys.path.insert(0,str(OLD_DIR))
import portfolio as previous
import mining as legacy
import slicing as old_slicing
OUT=ROOT/'artifacts/codepy_branch_guided_20261101'
