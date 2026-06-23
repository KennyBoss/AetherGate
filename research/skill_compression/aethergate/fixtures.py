"""Task source (today). Generates a REAL multi-file Python package on disk with a
failing pytest, and emits a TaskSpec for it. Replacing this module with a git
clone + patch-space author turns AetherGate loose on real GitHub repos — the
TaskSpec contract downstream does not change."""
from __future__ import annotations

import random
from pathlib import Path

from .taskspec import TaskSpec, Candidate


def _grid(*axes):
    out = [()]
    for ax in axes:
        out = [t + (v,) for t in out for v in ax]
    return out


def _src_add(p):
    o1, o2, c = p
    return f"def fn(x, y):\n    return (x {o1} y) {o2} {c}\n"


def _src_scale(p):
    a, b, o = p
    return f"def fn(x, y):\n    return {a} * x {o} {b} * y\n"


def _src_sel(p):
    (cmp,) = p
    return f"def fn(x, y):\n    return x if x {cmp} y else y\n"


MODULES = {
    "adder": (_src_add, _grid(["+", "-", "*"], ["+", "-", "*"], [0, 1, 2, 3])),
    "scaler": (_src_scale, _grid([1, 2, 3], [1, 2, 3], ["+", "-"])),
    "selector": (_src_sel, _grid(["<", "<=", ">", ">="])),
}
MOD_NAMES = list(MODULES)
TEST_INPUTS = [(2, 7), (6, 1), (3, 4), (9, 0), (5, 5)]


def _eval(mod, params):
    ns: dict = {}
    exec(MODULES[mod][0](params), ns)
    return [ns["fn"](x, y) for (x, y) in TEST_INPUTS]


def build_library(lib_seed: int, per_module: int) -> dict:
    r = random.Random(lib_seed)
    return {m: r.sample(grid, min(per_module, max(1, len(grid) // 2)))
            for m, (_s, grid) in MODULES.items()}


def make_task(out_root: Path, task_id: str, rng, library, kind: str,
              per_module: int) -> TaskSpec:
    buggy = rng.choice(MOD_NAMES)
    grid = MODULES[buggy][1]
    if kind == "reuse":
        correct = rng.choice(library[buggy])
    else:
        correct = rng.choice([p for p in grid if p not in set(library[buggy])])
    # other modules: correct params; expected outputs baked into the test
    correct_all = {m: (correct if m == buggy else library[m][0]) for m in MOD_NAMES}
    expected = {m: _eval(m, correct_all[m]) for m in MOD_NAMES}

    repo = out_root / task_id / "repo"
    (repo / "proj").mkdir(parents=True, exist_ok=True)
    (repo / "proj" / "__init__.py").write_text("", encoding="utf-8")
    for m in MOD_NAMES:
        # the buggy module ships BROKEN (a different param) so the test fails
        ship = correct_all[m]
        if m == buggy:
            ship = rng.choice([p for p in grid if p != correct])
        (repo / "proj" / f"{m}.py").write_text(MODULES[m][0](ship), encoding="utf-8")
    lines = [f"from proj.{m} import fn as {m}_fn" for m in MOD_NAMES]
    for m in MOD_NAMES:
        lines.append(f"def test_{m}():")
        for (x, y), e in zip(TEST_INPUTS, expected[m]):
            lines.append(f"    assert {m}_fn({x}, {y}) == {e!r}")
    (repo / "test_proj.py").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lib_set = set(library[buggy])
    patch_space = [
        Candidate(files={f"proj/{buggy}.py": MODULES[buggy][0](p)},
                  label="".join(map(str, p)), from_library=(p in lib_set))
        for p in grid]
    # CANONICAL order (by label), library NOT prioritised here — otherwise the
    # brute-force OFF baseline would unfairly inherit the skill ordering. Only the
    # APP / AetherGate arms reorder library-first / smoke-first, on purpose.
    patch_space.sort(key=lambda c: c.label)

    PYTEST = ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    return TaskSpec(
        task_id=task_id, repo_dir=str(repo), buggy_path=f"proj/{buggy}.py",
        smoke_cmd=PYTEST + ["-k", f"test_{buggy}"], full_cmd=PYTEST,
        patch_space=patch_space,
        meta={"kind": kind, "buggy_module": buggy, "correct": "".join(map(str, correct))})
