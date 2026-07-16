#!/usr/bin/env python3
"""Capture Git metadata, dirty patch, and changed/untracked files privately."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .private_output import ensure_private_output
except ImportError:
    from private_output import ensure_private_output


def git(root: Path, *args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return result.stdout if binary else result.stdout.decode("utf-8", errors="replace")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def nul_paths(data: bytes) -> list[str]:
    return [os.fsdecode(item) for item in data.split(b"\0") if item]


def status_paths(data: bytes) -> tuple[list[str], list[str]]:
    records = data.split(b"\0")
    paths: list[str] = []
    status_records: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status = os.fsdecode(record[:2])
        path = os.fsdecode(record[3:])
        paths.append(path)
        status_records.append(f"{status} {path}")
        if status[:1] in {"R", "C"} and index < len(records):
            index += 1
    return paths, status_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    destination = args.destination_dir.resolve()
    ensure_private_output(destination / ".snapshot-boundary-check")
    destination.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = destination / f"worktree-{stamp}"
    if snapshot_dir.exists():
        raise SystemExit(f"refusing to overwrite snapshot: {snapshot_dir}")
    snapshot_dir.mkdir()

    status_raw = bytes(git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", binary=True))
    parsed_paths, status_records = status_paths(status_raw)
    paths = set(parsed_paths)
    ignored_raw = bytes(git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", binary=True))
    ignored_paths = sorted(nul_paths(ignored_raw))

    unstaged_patch = bytes(git(root, "diff", "--binary", binary=True))
    staged_patch = bytes(git(root, "diff", "--cached", "--binary", binary=True))
    (snapshot_dir / "unstaged.patch").write_bytes(unstaged_patch)
    (snapshot_dir / "staged.patch").write_bytes(staged_patch)
    archive_path = snapshot_dir / "changed-and-untracked.zip"
    archived: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in sorted(paths):
            source = root / rel
            if not source.is_file():
                continue
            data = source.read_bytes()
            archive.writestr(rel, data)
            archived.append({"path": rel, "size_bytes": len(data), "sha256": sha256_bytes(data)})

    metadata = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": str(git(root, "branch", "--show-current")).strip(),
        "head": str(git(root, "rev-parse", "HEAD")).strip(),
        "commit_count": int(str(git(root, "rev-list", "--count", "HEAD")).strip()),
        "status_porcelain": status_records,
        "unstaged_patch_sha256": sha256_bytes(unstaged_patch),
        "staged_patch_sha256": sha256_bytes(staged_patch),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "archived_files": archived,
        "ignored_inventory": ignored_paths,
        "ignored_files_archived": False,
        "ignored_strategy": "Inventory only. Runtime databases use backup_sqlite.py; credentials and other ignored data require a dedicated private backup and are never copied into this archive implicitly.",
    }
    (snapshot_dir / "git-state.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"snapshot": str(snapshot_dir), "branch": metadata["branch"], "head": metadata["head"], "archived_files": len(archived)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
