"""The contract. A TaskSpec fully describes one repair task, independent of how
it was produced (generated fixture or cloned GitHub repo) or executed."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Candidate:
    """One candidate patch: file path(s) -> new full content. Generic over any
    repo (you provide the edited file contents)."""
    files: dict[str, str]
    label: str = ""
    from_library: bool = False     # is this a known fix-pattern (a "skill")?


@dataclass
class TaskSpec:
    task_id: str
    repo_dir: str                  # base repo on disk (has the failing test)
    buggy_path: str                # relative path the patches edit
    smoke_cmd: list[str]           # fast subset (e.g. pytest -k test_x)
    full_cmd: list[str]            # full suite (the expensive CI run)
    patch_space: list[Candidate]   # ordered candidate patches
    meta: dict = field(default_factory=dict)   # category, reuse/ood label, ...

    # ----- serialization (the JSON contract) -----
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        cands = [Candidate(**c) for c in d["patch_space"]]
        return TaskSpec(
            task_id=d["task_id"], repo_dir=d["repo_dir"], buggy_path=d["buggy_path"],
            smoke_cmd=list(d["smoke_cmd"]), full_cmd=list(d["full_cmd"]),
            patch_space=cands, meta=dict(d.get("meta", {})))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TaskSpec":
        return TaskSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def validate(self) -> None:
        assert Path(self.repo_dir).is_dir(), f"repo_dir missing: {self.repo_dir}"
        assert self.smoke_cmd and self.full_cmd, "smoke_cmd and full_cmd required"
        assert self.patch_space, "patch_space must be non-empty"

    @property
    def library(self) -> list[Candidate]:
        return [c for c in self.patch_space if c.from_library]
