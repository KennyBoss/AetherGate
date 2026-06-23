"""AetherGate — a smoke→(gate)→full cascade that cuts CI cost on code repair.

Production scaffold decoupled by a typed TaskSpec contract so the agent
(`engine.AetherGate`), the executor (`runner.IsolatedRunner`) and the task source
(`fixtures` today, real cloned repos tomorrow) evolve independently.
"""
from .taskspec import TaskSpec, Candidate
from .runner import IsolatedRunner, RunResult
from .engine import AetherGate

__all__ = ["TaskSpec", "Candidate", "IsolatedRunner", "RunResult", "AetherGate"]
