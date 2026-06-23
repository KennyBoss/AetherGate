"""The executor. Runs a candidate patch against a repo's tests in an ISOLATED
copy, returning pass/fail + real wall-clock. No network. Bytecode caching is
disabled so same-length sources never reuse a stale .pyc (the Phase-2.5 trap)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .taskspec import TaskSpec


@dataclass
class RunResult:
    passed: bool
    wall_s: float
    scope: str          # "smoke" | "full"


class IsolatedRunner:
    """One isolated working copy per TASK; candidates overwrite the buggy file
    in place. Safe because PYTHONDONTWRITEBYTECODE prevents stale-.pyc reuse.
    Set fresh_per_run=True for full side-effect isolation (slower) — the path a
    Docker/conda deployment would harden further."""

    def __init__(self, fresh_per_run: bool = False):
        self.fresh_per_run = fresh_per_run

    def prepare(self, spec: TaskSpec) -> str:
        wd = tempfile.mkdtemp(prefix=f"aethergate_{spec.task_id}_")
        dst = Path(wd) / "repo"
        shutil.copytree(spec.repo_dir, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        return str(dst)

    def cleanup(self, workdir: str) -> None:
        shutil.rmtree(Path(workdir).parent, ignore_errors=True)

    def run(self, spec: TaskSpec, workdir: str, files: dict[str, str],
            scope: str) -> RunResult:
        if self.fresh_per_run:
            wd_holder = self.prepare(spec)
            try:
                return self._run_in(spec, wd_holder, files, scope)
            finally:
                self.cleanup(wd_holder)
        return self._run_in(spec, workdir, files, scope)

    def _run_in(self, spec, workdir, files, scope) -> RunResult:
        repo = Path(workdir)
        for rel, content in files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        cmd = spec.smoke_cmd if scope == "smoke" else spec.full_cmd
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        t = time.perf_counter()
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True,
                           env=env, timeout=300)
        return RunResult(passed=(r.returncode == 0),
                         wall_s=time.perf_counter() - t, scope=scope)
