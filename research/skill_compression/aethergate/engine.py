"""The agent. The smoke -> (gate) -> full cascade and its baselines, all over the
TaskSpec contract. Logs the metrics a CTO reads: full_runs, smoke_runs, fire,
ci_seconds, and the saved-CI projection."""
from __future__ import annotations

from dataclasses import dataclass, field

from .taskspec import TaskSpec, Candidate
from .runner import IsolatedRunner


@dataclass
class RepairMetrics:
    arm: str
    solved: bool
    full_runs: int = 0
    smoke_runs: int = 0
    full_seconds: float = 0.0
    smoke_seconds: float = 0.0
    fired: bool = False        # did the cascade engage the skill library?
    fix_label: str = ""

    @property
    def ci_seconds(self) -> float:
        return self.full_seconds + self.smoke_seconds


class AetherGate:
    """smoke -> full cascade. The learned gate is an optional meta-decision
    (`gate_fn(spec)->bool`) of WHETHER to smoke-filter at all; default always on,
    which is the robust product mechanism."""

    def __init__(self, runner: IsolatedRunner, gate_fn=None):
        self.runner = runner
        self.gate_fn = gate_fn

    # ---- the product arm: smoke-filter then full ----
    def repair(self, spec: TaskSpec) -> RepairMetrics:
        wd = self.runner.prepare(spec)
        try:
            m = RepairMetrics(arm="aethergate", solved=False)
            engage = True if self.gate_fn is None else bool(self.gate_fn(spec))
            m.fired = engage
            survivors: list[Candidate] = []
            if engage:
                for c in spec.library:                      # cheap smoke filter
                    r = self.runner.run(spec, wd, c.files, "smoke")
                    m.smoke_runs += 1
                    m.smoke_seconds += r.wall_s
                    if r.passed:
                        survivors.append(c)
            order = survivors + [c for c in spec.patch_space if c not in survivors]
            for c in order:                                  # full only on chosen
                r = self.runner.run(spec, wd, c.files, "full")
                m.full_runs += 1
                m.full_seconds += r.wall_s
                if r.passed:
                    m.solved = True
                    m.fix_label = c.label
                    return m
            return m
        finally:
            self.runner.cleanup(wd)

    # ---- baselines ----
    def baseline_app(self, spec: TaskSpec) -> RepairMetrics:
        """Blind probe: full-test library first, then the rest."""
        wd = self.runner.prepare(spec)
        try:
            m = RepairMetrics(arm="app", solved=False)
            lib = set(id(c) for c in spec.library)
            order = spec.library + [c for c in spec.patch_space if id(c) not in lib]
            for c in order:
                r = self.runner.run(spec, wd, c.files, "full")
                m.full_runs += 1
                m.full_seconds += r.wall_s
                if r.passed:
                    m.solved = True
                    m.fix_label = c.label
                    return m
            return m
        finally:
            self.runner.cleanup(wd)

    def baseline_off(self, spec: TaskSpec) -> RepairMetrics:
        """Brute force: full-test the whole patch space in order."""
        wd = self.runner.prepare(spec)
        try:
            m = RepairMetrics(arm="off", solved=False)
            for c in spec.patch_space:
                r = self.runner.run(spec, wd, c.files, "full")
                m.full_runs += 1
                m.full_seconds += r.wall_s
                if r.passed:
                    m.solved = True
                    m.fix_label = c.label
                    return m
            return m
        finally:
            self.runner.cleanup(wd)
