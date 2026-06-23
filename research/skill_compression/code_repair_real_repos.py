#!/usr/bin/env python3
"""Phase 2.5 — does the fire-rate STRUCTURE survive heterogeneity + real pytest?

The riskiest test of the whole line: take the act/think gate out of the single
template it was tuned on and into a heterogeneous repair benchmark with many bug
classes, then measure the structure that survived MQAR -> compression ->
applicability -> hierarchy -> code-repair:

        fire_reuse  >  fire_mixed  >  fire_OOD

If that ordering holds across diverse function families AND under REAL `pytest`
execution, the law is observable outside the environment it was discovered in.

Scope honesty: these are GENERATED-but-REAL Python modules (real source, real
`pytest`), self-contained and dependency-free, so the sandbox stays confound-free
(no broken installs, no flaky network). Fetching actual GitHub repos is the next
step and needs network + dependency isolation; it is explicitly out of scope here.

Five heterogeneous function families (different patch-space sizes)
-----------------------------------------------------------------
  arith2 (36) | cmp (20) | linear (18) | select (4) | offbyone (6)
Each bug = wrong parameters; a repair finds parameters passing the tests. A
hidden library of recurring (family, params) fix patterns is the skill set.
  reuse bug: correct params in the library     -> a skill fixes it fast.
  OOD bug  : correct params not in the library -> the skill wastes test runs.

Arms (cost = test runs): OFF broad / APP probe / SSM learned gate.
Real-pytest layer: a sample of repairs is run end-to-end through actual `pytest`
subprocesses to (a) prove the loop works on real tests, (b) measure wall-clock per
run, (c) project the CI-seconds saved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    half = t * std / math.sqrt(n) if n > 1 else float("nan")
    return mean, std, half


# ---------------------------------------------------------------------------
# Heterogeneous function families: each renders REAL Python source from params.
# ---------------------------------------------------------------------------
def _arith2(p):
    o1, o2, c = p
    return f"def f(x, y):\n    return (x {o1} y) {o2} {c}"


def _cmp(p):
    cmp, c = p
    return f"def f(x, y):\n    return x {cmp} {c}"


def _linear(p):
    a, b, o = p
    return f"def f(x, y):\n    return {a} * x {o} {b} * y"


def _select(p):
    (cmp,) = p
    return f"def f(x, y):\n    return x if x {cmp} y else y"


def _offbyone(p):
    o1, o2 = p
    return f"def f(x, y):\n    return x {o1} (y {o2} 1)"


def _grid(*axes):
    out = [()]
    for ax in axes:
        out = [t + (v,) for t in out for v in ax]
    return out


FAMILIES = {
    "arith2": (_arith2, _grid(["+", "-", "*"], ["+", "-", "*"], [0, 1, 2, 3])),
    "cmp": (_cmp, _grid(["<", "<=", ">", ">=", "=="], [0, 1, 2, 3])),
    "linear": (_linear, _grid([1, 2, 3], [1, 2, 3], ["+", "-"])),
    "select": (_select, _grid(["<", "<=", ">", ">="])),
    "offbyone": (_offbyone, _grid(["+", "-", "*"], ["+", "-"])),
}


def compile_fn(fam, params):
    ns: dict = {}
    exec(FAMILIES[fam][0](params), ns)  # real code execution
    return ns["f"]


def passes(fam, params, cases, ref, upto=None):
    """Run candidate on cases; return count of matching outputs (one test run)."""
    f = compile_fn(fam, params)
    n = len(cases) if upto is None else min(upto, len(cases))
    ok = 0
    for i in range(n):
        x, y = cases[i]
        try:
            ok += int(f(x, y) == ref[i])
        except Exception:
            pass
    return ok


def make_bug(rng, fam, correct, k):
    cases = [(rng.randint(0, 9), rng.randint(0, 9)) for _ in range(k)]
    f = compile_fn(fam, correct)
    ref = [f(x, y) for (x, y) in cases]
    return {"fam": fam, "correct": correct, "cases": cases, "ref": ref}


# ---------------------------------------------------------------------------
# Repair (counts TEST RUNS)
# ---------------------------------------------------------------------------
def repair(bug, order):
    k = len(bug["cases"])
    for i, p in enumerate(order, start=1):
        if passes(bug["fam"], p, bug["cases"], bug["ref"]) == k:
            return i, True
    return len(order), False


def repair_app(bug, lib):
    runs1, ok = repair(bug, lib)
    if ok:
        return runs1, True
    rest = [p for p in FAMILIES[bug["fam"]][1] if p not in set(lib)]
    runs2, ok2 = repair(bug, rest)
    return runs1 + runs2, ok2


def repair_broad(bug):
    return repair(bug, FAMILIES[bug["fam"]][1])


def features(bug, lib, probe_cases):
    rows = []
    for p in lib:
        ok = passes(bug["fam"], p, bug["cases"], bug["ref"], upto=probe_cases)
        rows.append([ok / max(1, probe_cases), 1.0 if ok == probe_cases else 0.0])
    if not rows:
        rows = [[0.0, 0.0]]
    return np.asarray(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
def build_library(lib_seed, per_family):
    lib_rng = random.Random(lib_seed)
    lib = {}
    for fam, (_r, grid) in FAMILIES.items():
        n = min(per_family, max(1, len(grid) // 2))
        lib[fam] = lib_rng.sample(grid, n)
    return lib


def run_seed(seed, cfg):
    rng = random.Random(seed)
    lib = build_library(cfg.lib_seed, cfg.per_family)
    fams = list(FAMILIES)
    k = cfg.test_cases

    def gen_reuse():
        fam = rng.choice(fams)
        return make_bug(rng, fam, rng.choice(lib[fam]), k)

    def gen_ood():
        fam = rng.choice(fams)
        pool = [p for p in FAMILIES[fam][1] if p not in set(lib[fam])]
        return make_bug(rng, fam, rng.choice(pool), k)

    def fcost(bug):
        return len(lib[bug["fam"]]) * cfg.probe_cases / max(1, k)

    # gate training on own mixed experience
    gate_bugs = [gen_reuse() for _ in range(cfg.gate_train // 2)] + \
                [gen_ood() for _ in range(cfg.gate_train // 2)]
    rng.shuffle(gate_bugs)
    Xs, ys = [], []
    for b in gate_bugs:
        c_eng, _ = repair_app(b, lib[b["fam"]])
        c_broad, _ = repair_broad(b)
        ys.append(1 if c_eng < c_broad else 0)
        Xs.append(features(b, lib[b["fam"]], cfg.probe_cases))

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

    eval_reuse = [gen_reuse() for _ in range(cfg.eval_bugs)]
    eval_ood = [gen_ood() for _ in range(cfg.eval_bugs)]

    def measure(bugs):
        off = [repair_broad(b)[0] for b in bugs]
        app = [repair_app(b, lib[b["fam"]])[0] for b in bugs]
        ssm, fired = [], 0
        for b in bugs:
            if gate.predict(features(b, lib[b["fam"]], cfg.probe_cases)) >= 0.5:
                fired += 1
                r, _ = repair_app(b, lib[b["fam"]])
            else:
                r, _ = repair_broad(b)
            ssm.append(r + fcost(b))
        return {"off": statistics.mean(off), "base_app": statistics.mean(app),
                "base_ssm": statistics.mean(ssm), "fire_rate": fired / len(bugs)}

    out = {"seed": seed, "gate_acc": round(acc, 3)}
    for split, bugs in (("reuse", eval_reuse), ("ood", eval_ood)):
        out[split] = measure(bugs)
    out["mixed"] = {
        kk: (out["reuse"][kk] + out["ood"][kk]) / 2
        for kk in ("off", "base_app", "base_ssm", "fire_rate")}
    return out


# ---------------------------------------------------------------------------
# Real pytest layer: run a repair end-to-end through actual pytest subprocesses.
# ---------------------------------------------------------------------------
_PYTEST_RUN_ID = [0]


def pytest_run(workdir, fam, params, cases, ref) -> bool:
    # Unique subdir per run + no bytecode cache: same-length sources written in
    # the same second otherwise collide on mtime+size and reuse a STALE .pyc,
    # making pytest execute the previous candidate. Both guards make it correct.
    _PYTEST_RUN_ID[0] += 1
    rundir = workdir / f"r{_PYTEST_RUN_ID[0]}"
    rundir.mkdir(parents=True, exist_ok=True)
    src = FAMILIES[fam][0](params)
    (rundir / "bug_mod.py").write_text(src + "\n", encoding="utf-8")
    lines = ["from bug_mod import f", "def test_f():"]
    for (x, y), r in zip(cases, ref):
        lines.append(f"    assert f({x}, {y}) == {r!r}")
    (rundir / "test_bug.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(rundir)], capture_output=True, text=True, timeout=120, env=env)
    return r.returncode == 0


def real_pytest_calibration(cfg, n_bugs=8):
    rng = random.Random(99)
    lib = build_library(cfg.lib_seed, cfg.per_family)
    fams = list(FAMILIES)
    runs, wall, agree = 0, 0.0, 0
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for _ in range(n_bugs):
            fam = rng.choice(fams)
            correct = rng.choice(lib[fam])
            bug = make_bug(rng, fam, correct, cfg.test_cases)
            for p in FAMILIES[fam][1]:
                t = time.perf_counter()
                ok = pytest_run(wd, fam, p, bug["cases"], bug["ref"])
                wall += time.perf_counter() - t
                runs += 1
                inproc = passes(fam, p, bug["cases"], bug["ref"]) == cfg.test_cases
                agree += int(ok == inproc)
                if ok:
                    break
    return {"pytest_runs": runs, "total_wall_s": round(wall, 2),
            "wall_per_run_s": round(wall / max(1, runs), 4),
            "pass_fail_agreement": round(agree / max(1, runs), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="11,17,23,41,73,101")
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--lib-seed", type=int, default=777)
    ap.add_argument("--test-cases", type=int, default=5)
    ap.add_argument("--probe-cases", type=int, default=2)
    ap.add_argument("--gate-train", type=int, default=160)
    ap.add_argument("--eval-bugs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--real-pytest-sample", type=int, default=8)
    ap.add_argument("--output-json", default="artifacts/code_repair_real_repos.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = [run_seed(s, args) for s in seeds]

    summary = {}
    for split in ("reuse", "mixed", "ood"):
        def m(key):
            return round(statistics.mean(r[split][key] for r in runs), 3)
        summary[split] = {"off": m("off"), "app": m("base_app"),
                          "ssm": m("base_ssm"), "fire_rate": m("fire_rate")}
    gate_acc = round(statistics.mean(r["gate_acc"] for r in runs), 3)

    # the structure test
    fr = {s: [r[s]["fire_rate"] for r in runs] for s in ("reuse", "mixed", "ood")}
    structure_per_seed = [fr["reuse"][i] > fr["mixed"][i] > fr["ood"][i]
                          for i in range(len(seeds))]
    structure_holds = all(structure_per_seed)
    fr_reuse_m, _, fr_reuse_h = ci95(fr["reuse"])
    fr_ood_m, _, fr_ood_h = ci95(fr["ood"])

    # product metric: test-run savings + real CI-seconds projection
    cal = real_pytest_calibration(args, args.real_pytest_sample)
    wps = cal["wall_per_run_s"]
    off_mix, app_mix, ssm_mix = summary["mixed"]["off"], summary["mixed"]["app"], summary["mixed"]["ssm"]
    savings = {
        "ssm_vs_off_runs_pct": round(100 * (off_mix - ssm_mix) / off_mix, 1),
        "ssm_vs_app_runs_pct": round(100 * (app_mix - ssm_mix) / app_mix, 1),
        "ci_seconds_saved_per_bug_ssm_vs_off": round((off_mix - ssm_mix) * wps, 3),
    }

    payload = {"config": vars(args), "gate_acc": gate_acc, "summary": summary,
               "fire_structure": {
                   "ordering": "fire_reuse > fire_mixed > fire_ood",
                   "holds_all_seeds": structure_holds,
                   "per_seed": structure_per_seed,
                   "fire_reuse_mean_ci95": [round(fr_reuse_m, 3), round(fr_reuse_h, 3)],
                   "fire_ood_mean_ci95": [round(fr_ood_m, 3), round(fr_ood_h, 3)]},
               "real_pytest_calibration": cal, "ci_savings": savings,
               "seed_runs": runs,
               "protocol_evidence": {"heterogeneous_families": list(FAMILIES),
                                     "real_pytest_subprocess": True,
                                     "n_seeds": len(seeds), "valid": len(seeds) >= 3}}
    out = Path(args.output_json)
    if not out.is_absolute():
        out = Path(__file__).parent / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== Phase 2.5 — heterogeneous repair + real pytest ===")
    print(f"seeds={seeds}  families={list(FAMILIES)}  gate_acc={gate_acc}")
    print(f"  test runs to fix (cost):  {'OFF':>7s} {'APP':>7s} {'SSM':>7s} {'fire':>7s}")
    for split in ("reuse", "mixed", "ood"):
        s = summary[split]
        print(f"    {split:7s}              {s['off']:7.2f} {s['app']:7.2f} "
              f"{s['ssm']:7.2f} {s['fire_rate']:7.3f}")
    print(f"  STRUCTURE  fire_reuse > fire_mixed > fire_ood :")
    print(f"    reuse={summary['reuse']['fire_rate']}  mixed={summary['mixed']['fire_rate']}"
          f"  ood={summary['ood']['fire_rate']}  -> holds_all_seeds={structure_holds}")
    print(f"  REAL PYTEST: {cal['pytest_runs']} runs, {cal['wall_per_run_s']}s/run, "
          f"pass/fail agreement with in-proc = {cal['pass_fail_agreement']}")
    print(f"  CI SAVINGS (mixed): SSM uses {savings['ssm_vs_off_runs_pct']}% fewer "
          f"runs than OFF; {savings['ci_seconds_saved_per_bug_ssm_vs_off']}s CI saved/bug")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
