#!/usr/bin/env python3
"""Fail when a proposed public tree contains private runtime/context paths."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def normalize(path: str) -> str:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def denied(path: str) -> bool:
    normalized = normalize(path)
    parts = tuple(part for part in normalized.split("/") if part)
    if not parts:
        return False
    lowered = tuple(part.lower() for part in parts)
    name = lowered[-1]

    if ".agents" in lowered:
        return True
    if any(part.startswith(".private_") for part in lowered):
        return True
    if any(part.startswith("security_hardening_") for part in lowered):
        return True
    if "reports" in lowered or "data" in lowered:
        return True
    if normalized == "brain_index.json":
        return True
    if len(parts) == 2 and lowered[0] == "memory" and name.endswith((".json", ".jsonl")):
        return True

    if name in {".env.example", "config.example.json"}:
        return False
    if name == ".env" or name.startswith(".env."):
        return True
    if name in {"config.json", "config.local.json", "config.private.json"}:
        return True

    if name.endswith((".sqlite", ".sqlite3", ".db", ".db3")):
        return True
    if any(marker in name for marker in (".sqlite-", ".sqlite3-", ".db-", ".db3-")):
        return True
    if name.endswith(("-wal", "-shm", "-journal", ".wal", ".shm", ".journal")):
        return True
    if name.endswith((".dump", ".sql.gz", ".log", ".bundle")):
        return True
    if ".bak" in name or ".backup" in name:
        return True
    return False


def tracked_paths(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True, encoding="utf-8")
    return [line for line in output.splitlines() if line]


def tree_paths(root: Path) -> list[str]:
    return [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--all-files", action="store_true", help="scan every file instead of Git tracked paths")
    args = parser.parse_args()
    root = args.root.resolve()
    paths = tree_paths(root) if args.all_files else tracked_paths(root)
    findings = sorted(path for path in paths if denied(path))
    for path in findings:
        print(f"DENY path={path}")
    print(f"checked={len(paths)} denied={len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
