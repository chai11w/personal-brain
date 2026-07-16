"""Enforce ignored output targets for private artifacts inside Git worktrees."""

from __future__ import annotations

import subprocess
from pathlib import Path


def _existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _git_root(path: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(_existing_ancestor(path)), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def ensure_private_output(path: Path) -> None:
    """Reject an output path inside a Git worktree unless Git ignores it."""
    resolved = path.resolve()
    root = _git_root(resolved)
    if root is None:
        return
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError:
        return
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--no-index", "-q", "--", relative],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"refusing private artifact in non-ignored Git path: {relative}")
