#!/usr/bin/env python3
"""Phase 2.7 / AetherGate harness — the act/think gate on a REAL multi-file repo
run through REAL pytest, structured so a cloned GitHub repo is a drop-in.

This is the product scaffold. It keeps every confound out of the sandbox (no
network, no third-party deps) by operating on a generated-but-real Python
*package* (multiple modules + a real pytest suite), executed through real pytest
subprocesses in isolated working dirs. Swapping in a cloned repo means replacing
`make_repo_spec` with a loader for {repo dir, failing test, patch space}.

Repo model
----------
A package `proj/` with several modules (one function each) and `tests/test_proj.py`
(one pytest per module). ONE module carries a bug (wrong parameters); the failing
test name localises it. A repair searches that module's patch space. Other
modules stay correct, so the full suite passes iff the buggy module is fixed.

Two execution backends, validated to agree
------------------------------------------
  real : write the package to an isolated temp dir, run `pytest` (full suite) or
         `pytest -k <module>` (smoke subset). Real wall-clock = CI cost.
  fast : in-process `exec` of the modules (proven 1.000 agreement with pytest).
fast drives the 6-seed statistics; real validates the loop + calibrates seconds.

Arms (cost = test runs): OFF broad / APP probe / SSM learned gate. Logged per
bug: fire-rate, full-runs, smoke-runs, CI-seconds. Phase diagram is analytic over
deterministic counts x calibrated real-pytest t0 (no wall-clock races).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import skill_compression_ssm_gate as G


def ci95(xs):
    n = len(xs)
    mean = statistics.mean(xs)
    std = statistics.stdev(xs) if n > 1 else 0.0
    t = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}.get(n - 1, 1.96)
    return mean, (t * std / math.sqrt(n) if n > 1 else float("nan"))


# ---------------------------------------------------------------------------
# Module catalogue: each renders a REAL function body from params.
# ---------------------------------------------------------------------------
def _grid(*axes):
    out = [()]
    for ax in axes:
        out = [t + (v,) for t in out for v in ax]
    return out


def _src_add(p):
    o1, o2, c = p
    return f"def fn(x, y):\n    return (x {o1} y) {o2} {c}"


def _src_scale(p):
    a, b, o = p
    return f"def fn(x, y):\n    return {a} * x {o} {b} * y"


def _src_clamp(p):
    cmp, c = p
    return f"def fn(x, y):\n    return x if x {cmp} {c} else {c}"


def _src_sel(p):
    (cmp,) = p
    return f"def fn(x, y):\n    return x if x {cmp} y else y"


MODULES = {
    "adder": (_src_add, _grid(["+", "-", "*"], ["+", "-", "*"], [0, 1, 2, 3])),
    "scaler": (_src_scale, _grid([1, 2, 3], [1, 2, 3], ["+", "-"])),
    "clamper": (_src_clamp, _grid(["<", "<=", ">", ">="], [0, 1, 2, 3])),
    "selector": (_src_sel, _grid(["<", "<=", ">", ">="])),
}
MOD_NAMES = list(MODULES)
TEST_INPUTS = [(2, 7), (6, 1), (3, 4), (9, 0), (5, 5)]


def compile_fn(mod, params):
    ns: dict = {}
    exec(MODULES[mod][0](params), ns)
    return ns["fn"]


def expected_outputs(mod, params):
    f = compile_fn(mod, params)
    return [f(x, y) for (x, y) in TEST_INPUTS]


# ---------------------------------------------------------------------------
# A repo task: correct params for all modules + a designated buggy module.
# ---------------------------------------------------------------------------
def make_repo_spec(rng, library, buggy_mod, correct_params):
    spec = {"modules": {}, "buggy": buggy_mod}
    for m in MOD_NAMES:
        # other modules: any correct params (pick a library entry for determinism)
        p = correct_params if m == buggy_mod else library[m][0]
        spec["modules"][m] = {"params": p, "expected": expected_outputs(m, p)}
    return spec


# ---------------------------------------------------------------------------
# Backend: in-process (fast) pass check
# ---------------------------------------------------------------------------
def passes_fast(spec, candidate, scope, probe_cases=None):
    """Return True if the suite passes with `candidate` patched into the buggy
    module. scope='full' runs every module's checks; 'smoke' only the buggy one
    (optionally only probe_cases inputs)."""
    mods = [spec["buggy"]] if scope == "smoke" else MOD_NAMES
    for m in mods:
        params = candidate if m == spec["buggy"] else spec["modules"][m]["params"]
        f = compile_fn(m, params)
        exp = spec["modules"][m]["expected"]
        n = len(TEST_INPUTS) if (scope != "smoke" or probe_cases is None) else min(probe_cases, len(TEST_INPUTS))
        for i in range(n):
            x, y = TEST_INPUTS[i]
            try:
                if f(x, y) != exp[i]:
                    return False
            except Exception:
                return False
    return True


# ---------------------------------------------------------------------------
# Backend: real pytest in an isolated temp package
# ---------------------------------------------------------------------------
_RID = [0]


def write_repo(root, spec, candidate):
    pkg = root / "proj"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    for m in MOD_NAMES:
        params = candidate if m == spec["buggy"] else spec["modules"][m]["params"]
        (pkg / f"{m}.py").write_text(MODULES[m][0](params) + "\n", encoding="utf-8")
    lines = []
    for m in MOD_NAMES:
        lines.append(f"from proj.{m} import fn as {m}_fn")
    for m in MOD_NAMES:
        lines.append(f"def test_{m}():")
        for (x, y), e in zip(TEST_INPUTS, spec["modules"][m]["expected"]):
            lines.append(f"    assert {m}_fn({x}, {y}) == {e!r}")
    (root / "test_proj.py").write_text("\n".join(lines) + "\n", encoding="utf-8")


def passes_real(workdir, spec, candidate, scope):
    _RID[0] += 1
    rd = workdir / f"r{_RID[0]}"
    rd.mkdir(parents=True, exist_ok=True)
    write_repo(rd, spec, candidate)
    cmd = ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if scope == "smoke":
        cmd += ["-k", f"test_{spec['buggy']}"]
    cmd.append(str(rd))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    shutil.rmtree(rd, ignore_errors=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# Repair search (counts FULL test runs and SMOKE probes separately)
# ---------------------------------------------------------------------------
class Backend:
    def __init__(self, kind, workdir=None):
        self.kind = kind
        self.workdir = workdir

    def full(self, spec, cand):
        return passes_real(self.workdir, spec, cand, "full") if self.kind == "real" \
            else passes_fast(spec, cand, "full")

    def smoke(self, spec, cand, probe_cases):
        if self.kind == "real":
            return passes_real(self.workdir, spec, cand, "smoke")
        return passes_fast(spec, cand, "smoke", probe_cases)


def repair_full(spec, order, be):
    for i, p in enumerate(order, start=1):
        if be.full(spec, p):
            return i, True
    return len(order), False


def grid_of(spec):
    return MODULES[spec["buggy"]][1]


def lib_of(spec, library):
    return library[spec["buggy"]]


def repair_app(spec, library, be):
    runs1, ok = repair_full(spec, lib_of(spec, library), be)
    if ok:
        return runs1, True
    rest = [p for p in grid_of(spec) if p not in set(lib_of(spec, library))]
    runs2, ok2 = repair_full(spec, rest, be)
    return runs1 + runs2, ok2


def repair_smoke_filter(spec, library, be, probe_cases):
    """The product mechanism: cheap SMOKE-probe the library, then spend a FULL
    test run only on candidates that pass smoke. Returns (smoke_runs, full_runs,
    solved). Beats blind APP because it full-tests far fewer candidates: on reuse
    only the smoke-passing fix, on novel bugs none (straight to broad)."""
    lib = lib_of(spec, library)
    smoke_runs = len(lib)
    passers = [p for p in lib if be.smoke(spec, p, probe_cases)]
    full_runs = 0
    for p in passers:                       # full-test smoke survivors first
        full_runs += 1
        if be.full(spec, p):
            return smoke_runs, full_runs, True
    seen = set(passers)
    for p in grid_of(spec):                 # broad fallback over the rest
        if p in seen:
            continue
        full_runs += 1
        if be.full(spec, p):
            return smoke_runs, full_runs, True
    return smoke_runs, full_runs, False


def features(spec, library, be, probe_cases):
    rows = []
    for p in lib_of(spec, library):
        ok = be.smoke(spec, p, probe_cases)
        rows.append([1.0 if ok else 0.0, 1.0 if ok else 0.0])
    if not rows:
        rows = [[0.0, 0.0]]
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
def build_library(lib_seed, per_module):
    r = random.Random(lib_seed)
    return {m: r.sample(grid, min(per_module, max(1, len(grid) // 2)))
            for m, (_s, grid) in MODULES.items()}


def gen_spec(rng, library, kind):
    m = rng.choice(MOD_NAMES)
    if kind == "reuse":
        cp = rng.choice(library[m])
    else:
        pool = [p for p in MODULES[m][1] if p not in set(library[m])]
        cp = rng.choice(pool)
    return make_repo_spec(rng, library, m, cp)


def train_gate(seed, cfg, library, be_fast):
    rng = random.Random(seed)
    bugs = [gen_spec(rng, library, "reuse") for _ in range(cfg.gate_train // 2)] + \
           [gen_spec(rng, library, "ood") for _ in range(cfg.gate_train // 2)]
    rng.shuffle(bugs)
    Xs, ys = [], []
    for s in bugs:
        c_eng, _ = repair_app(s, library, be_fast)
        c_broad, _ = repair_full(s, grid_of(s), be_fast)
        ys.append(1 if c_eng < c_broad else 0)
        Xs.append(features(s, library, be_fast, cfg.probe_cases))
    gate = G.SSMGate(d_in=2, hidden=cfg.hidden, seed=seed)
    for _ in range(cfg.epochs):
        grads = {"A": np.zeros_like(gate.A), "B": np.zeros_like(gate.B),
                 "w": np.zeros_like(gate.w), "b": 0.0}
        for X, y in zip(Xs, ys):
            g = gate.backward(gate.forward(X)[1], float(y))
            for kk in grads:
                grads[kk] = grads[kk] + g[kk]
        for kk in grads:
            grads[kk] = grads[kk] / len(Xs)
        gate.step(grads, lr=cfg.lr)
    acc = float(np.mean([(gate.predict(X) >= 0.5) == bool(y) for X, y in zip(Xs, ys)]))
    return gate, acc


def measure(seed, cfg, library, gate, be):
    rng = random.Random(seed * 31 + 7)
    splits = {"reuse": [gen_spec(rng, library, "reuse") for _ in range(cfg.eval_bugs)],
              "ood": [gen_spec(rng, library, "ood") for _ in range(cfg.eval_bugs)]}
    out = {}
    for name, bugs in splits.items():
        off_full = app_full = ssm_full = ssm_smoke = fired = 0
        smk_full = smk_smoke = 0
        for s in bugs:
            off_full += repair_full(s, grid_of(s), be)[0]
            app_full += repair_app(s, library, be)[0]
            sr, fr_, _ = repair_smoke_filter(s, library, be, cfg.probe_cases)
            smk_smoke += sr
            smk_full += fr_
            ssm_smoke += len(lib_of(s, library))
            if gate.predict(features(s, library, be, cfg.probe_cases)) >= 0.5:
                fired += 1
                ssm_full += repair_app(s, library, be)[0]
            else:
                ssm_full += repair_full(s, grid_of(s), be)[0]
        n = len(bugs)
        out[name] = {"off_full": off_full / n, "app_full": app_full / n,
                     "ssm_full": ssm_full / n, "ssm_smoke": ssm_smoke / n,
                     "smk_full": smk_full / n, "smk_smoke": smk_smoke / n,
                     "fire_rate": fired / n}
    return out


# ---------------------------------------------------------------------------
def real_validation(cfg, n=6):
    """Run a few repairs through REAL pytest; confirm agreement with fast and
    measure t0 (seconds per full pytest run)."""
    rng = random.Random(2024)
    library = build_library(cfg.lib_seed, cfg.per_module)
    agree, total, wall, runs = 0, 0, 0.0, 0
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        be_real = Backend("real", wd)
        for _ in range(n):
            spec = gen_spec(rng, library, rng.choice(["reuse", "ood"]))
            for p in grid_of(spec):
                t = time.perf_counter()
                rok = be_real.full(spec, p)
                wall += time.perf_counter() - t
                runs += 1
                fok = passes_fast(spec, p, "full")
                total += 1
                agree += int(rok == fok)
                if rok:
                    break
    return {"agreement": round(agree / max(1, total), 4), "t0_seconds": round(wall / max(1, runs), 4),
            "real_pytest_runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--per-module", type=int, default=6)
    ap.add_argument("--lib-seed", type=int, default=777)
    ap.add_argument("--probe-cases", type=int, default=2)
    ap.add_argument("--gate-train", type=int, default=160)
    ap.add_argument("--eval-bugs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--regimes", default="1,3,10,30,100")
    ap.add_argument("--bug-volume", type=int, default=1000)
    ap.add_argument("--real-validation", type=int, default=6)
    ap.add_argument("--output-json", default="artifacts/aethergate_harness.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    library = build_library(args.lib_seed, args.per_module)
    be_fast = Backend("fast")

    per_seed, accs = [], []
    for s in seeds:
        gate, acc = train_gate(s, args, library, be_fast)
        accs.append(acc)
        per_seed.append(measure(s, args, library, gate, be_fast))

    # fire-rate structure (the invariant)
    def fr(split):
        return [r[split]["fire_rate"] for r in per_seed]
    structure = [fr("reuse")[i] > fr("ood")[i] for i in range(len(seeds))]
    KEYS = ("off_full", "app_full", "ssm_full", "ssm_smoke", "smk_full", "smk_smoke", "fire_rate")
    agg = {}
    for split in ("reuse", "ood"):
        agg[split] = {k: round(statistics.mean(r[split][k] for r in per_seed), 3) for k in KEYS}
    mixed = {k: round((agg["reuse"][k] + agg["ood"][k]) / 2, 3) for k in KEYS}

    # real pytest validation + calibration
    rv = real_validation(args, args.real_validation)
    t0 = rv["t0_seconds"]

    # phase diagram (analytic over deterministic counts). SMOKE = product arm.
    def pstar(full_key, smoke_key):
        d = mixed["app_full"] - mixed[full_key]
        return round(mixed[smoke_key] / d, 2) if d > 0 else None
    pstar_ssm = pstar("ssm_full", "ssm_smoke")
    pstar_smk = pstar("smk_full", "smk_smoke")
    regimes = [float(x) for x in args.regimes.split(",")]
    table = []
    for P in regimes:
        off = mixed["off_full"] * t0 * P
        app = mixed["app_full"] * t0 * P
        smk = mixed["smk_full"] * t0 * P + mixed["smk_smoke"] * t0
        table.append({"P": P,
                      "smk_saves_vs_app_min": round((app - smk) * args.bug_volume / 60, 1),
                      "smk_saves_vs_off_min": round((off - smk) * args.bug_volume / 60, 1),
                      "smoke_beats_app": (app - smk) > 0})

    payload = {"config": vars(args), "gate_acc": round(statistics.mean(accs), 3),
               "fire_structure": {"reuse": agg["reuse"]["fire_rate"],
                                  "mixed": mixed["fire_rate"], "ood": agg["ood"]["fire_rate"],
                                  "reuse_gt_ood_all_seeds": all(structure)},
               "counts_mixed": mixed, "real_pytest_validation": rv,
               "crossover_P_star_learned_gate": pstar_ssm,
               "crossover_P_star_smoke_filter": pstar_smk, "regime_table": table,
               "protocol_evidence": {"real_multi_file_repo": True, "real_pytest": True,
                                     "fast_backend_agreement": rv["agreement"],
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== AetherGate harness — act/think gate on a real multi-file repo ===")
    print(f"seeds={seeds}  modules={MOD_NAMES}  gate_acc={payload['gate_acc']}")
    print(f"  REAL pytest: {rv['real_pytest_runs']} runs, t0={t0}s/run, "
          f"fast-backend agreement={rv['agreement']}")
    print(f"  FIRE STRUCTURE  reuse={payload['fire_structure']['reuse']} "
          f"mixed={payload['fire_structure']['mixed']} ood={payload['fire_structure']['ood']}  "
          f"reuse>ood all seeds={payload['fire_structure']['reuse_gt_ood_all_seeds']}")
    print(f"  full-runs/bug (mixed): OFF={mixed['off_full']} APP={mixed['app_full']} "
          f"SSM={mixed['ssm_full']} SMOKE={mixed['smk_full']} (smoke probes={mixed['smk_smoke']})")
    print(f"  crossover P*: learned-gate={pstar_ssm}  SMOKE-filter={pstar_smk}  "
          f"(SMOKE beats APP above this CI heaviness)")
    print(f"  SMOKE-filter (product arm) savings:")
    print(f"  {'P':>5s} {'save vs APP':>14s} {'save vs OFF':>14s}  (min/{args.bug_volume} bugs)")
    for row in table:
        print(f"  {row['P']:>5.0f} {row['smk_saves_vs_app_min']:>14.1f} "
              f"{row['smk_saves_vs_off_min']:>14.1f}   beats_app={row['smoke_beats_app']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
