"""Task source (tomorrow). Build a TaskSpec from a real repository — a local
checkout now, a git clone when there is network. Generates a candidate patch
space by single-token mutation of the buggy file (the "one wrong operator/constant"
bug class), so the loader works on arbitrary Python files, not just fixtures.

Local path is fully runnable here; `clone_repo` is the thin, network-gated wrapper
for real GitHub repos (https://github.com/KennyBoss/AetherGate)."""
from __future__ import annotations

import io
import subprocess
import tokenize
from pathlib import Path

from .taskspec import TaskSpec, Candidate

# single-token mutation tables (the "one wrong operator/constant" bug class)
OP_MUT = {
    "+": ["-", "*"], "-": ["+", "*"], "*": ["+", "-"], "//": ["*", "+"],
    "<": ["<=", ">", ">="], "<=": ["<", ">", ">="],
    ">": [">=", "<", "<="], ">=": [">", "<", "<="],
    "==": ["!="], "!=": ["=="],
}
INT_CHOICES = ["0", "1", "2", "3"]


def _token_sites(src: str):
    """Yield (start_offset, end_offset, text, alternatives) for mutable tokens."""
    lines = src.splitlines(keepends=True)
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))

    def pos(rc):
        r, c = rc
        return offs[r - 1] + c

    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in toks:
        if tok.type == tokenize.OP and tok.string in OP_MUT:
            yield pos(tok.start), pos(tok.end), tok.string, OP_MUT[tok.string]
        elif tok.type == tokenize.NUMBER and tok.string in INT_CHOICES:
            alts = [c for c in INT_CHOICES if c != tok.string]
            yield pos(tok.start), pos(tok.end), tok.string, alts


def mutation_patch_space(repo_dir, buggy_path, library_signatures=None):
    """Candidate file contents, each one single-token mutation of the buggy file.
    library_signatures: optional set of (orig_token, new_token) flagged as known
    fix-patterns (skills)."""
    src = (Path(repo_dir) / buggy_path).read_text(encoding="utf-8")
    library_signatures = set(library_signatures or [])
    out, seen = [], set()
    for a, b, text, alts in _token_sites(src):
        for nt in alts:
            patched = src[:a] + nt + src[b:]
            if patched in seen or patched == src:
                continue
            seen.add(patched)
            out.append(Candidate(
                files={buggy_path: patched},
                label=f"{a}:{text}->{nt}",
                from_library=((text, nt) in library_signatures)))
    return out


def build_taskspec_local(repo_dir, buggy_path, smoke_cmd, full_cmd, *,
                         task_id="local", library_signatures=None, meta=None):
    spec = TaskSpec(
        task_id=task_id, repo_dir=str(repo_dir), buggy_path=str(buggy_path),
        smoke_cmd=list(smoke_cmd), full_cmd=list(full_cmd),
        patch_space=mutation_patch_space(repo_dir, buggy_path, library_signatures),
        meta=dict(meta or {}))
    spec.validate()
    return spec


# -------------------------------------------------------------------------
# Network-gated: clone a real GitHub repo. Out of scope for the offline sandbox;
# the interface is fixed so deployment is a drop-in.
# -------------------------------------------------------------------------
def clone_repo(url, dest, commit=None, depth=1):
    dest = Path(dest)
    subprocess.run(["git", "clone", "--depth", str(depth), url, str(dest)], check=True)
    if commit:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", commit], check=True)
    return str(dest)


def build_taskspec_from_git(url, buggy_path, smoke_cmd, full_cmd, dest, *,
                            commit=None, task_id=None, library_signatures=None, meta=None):
    repo = clone_repo(url, dest, commit=commit)               # needs network
    return build_taskspec_local(repo, buggy_path, smoke_cmd, full_cmd,
                                task_id=task_id or url.rsplit("/", 1)[-1],
                                library_signatures=library_signatures, meta=meta)
