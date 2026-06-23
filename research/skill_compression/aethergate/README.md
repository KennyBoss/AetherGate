# AetherGate — a smoke→(gate)→full cascade that cuts CI cost on code repair

Production scaffold for the result of the skill-compression line: **run the cheap
test first; commit the expensive full CI run only on candidates that survive.**

## Architecture (decoupled by a typed contract)

```
TaskSpec  (the contract)      engine.AetherGate          runner.IsolatedRunner
 ├ repo_dir                    ├ repair()  smoke→full      ├ prepare() copytree
 ├ buggy_path                  ├ baseline_app()  (probe)   ├ run() pytest, isolated
 ├ smoke_cmd / full_cmd        └ baseline_off()  (brute)   └ cleanup()
 ├ patch_space[Candidate]
 └ meta
```

* **`taskspec.py`** — `TaskSpec` / `Candidate`, JSON-serializable. The contract
  that lets the agent, the executor and the task source evolve independently.
* **`runner.py`** — `IsolatedRunner` runs a candidate in an isolated repo copy via
  real `pytest` subprocess; `PYTHONDONTWRITEBYTECODE` kills the stale-`.pyc` trap;
  `fresh_per_run=True` gives full side-effect isolation (Docker hardens further).
* **`engine.py`** — the smoke→full cascade and its baselines, emitting
  `RepairMetrics` (full_runs, smoke_runs, ci_seconds).
* **`fixtures.py`** — generates *real* multi-file packages with failing `pytest`.
  Replace it with a git-clone + patch-space author to run on real GitHub repos;
  nothing downstream changes.

## Validated end-to-end (real isolated pytest, `run_aethergate_demo.py`)

Deterministic full-runs/bug across 12 tasks: **OFF 17.75 → APP 9.92 →
AetherGate 6.75** (+5 smoke). Crossover `P* ≈ 1.6`; vs the strong probe AetherGate
saves up to **+778 min/1k bugs**, vs brute force **+2733 min (~45 h)/1k** at heavy
CI. A sample contract is in `../artifacts/sample_taskspec.json`.

## To run on a real repo (`repo_loader.py`)

`repo_loader` builds a `TaskSpec` from any **local checkout** (auto patch space by
single-token mutation of the buggy file — the "one wrong operator/constant" class)
and from a **git clone** (network-gated, the only deployment add):

```python
from aethergate import IsolatedRunner, AetherGate
from aethergate.repo_loader import build_taskspec_local, build_taskspec_from_git

PYTEST = ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
spec = build_taskspec_local("/path/to/checkout", "pkg/buggy.py",
                            PYTEST + ["-k", "test_buggy"], PYTEST,
                            library_signatures={("-", "+")})
print(AetherGate(IsolatedRunner(fresh_per_run=True)).repair(spec))
# build_taskspec_from_git(url, buggy_path, smoke_cmd, full_cmd, dest)  # needs network
```

The only missing piece for real GitHub repos is environment isolation for
third-party deps (venv/conda/Docker) — out of scope for the confound-free sandbox,
a one-module add at deployment.
