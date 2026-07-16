#!/usr/bin/env python3
"""Conservative worktree scan that reports detector/path/line, never values."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DETECTORS = {
    "private_key_header": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "generic_secret_assignment": re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*[\"']?[A-Za-z0-9_./+\-=]{16,}"),
    "bearer_token": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"),
}


def tracked(root: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True, encoding="utf-8")
    return [root / line for line in output.splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings: list[tuple[str, str, int]] = []
    scanned = 0
    for path in tracked(root):
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        rel = path.relative_to(root).as_posix()
        for line_number, line in enumerate(lines, 1):
            for detector, pattern in DETECTORS.items():
                if pattern.search(line):
                    findings.append((detector, rel, line_number))
    for detector, rel, line_number in findings:
        print(f"FINDING detector={detector} path={rel} line={line_number}")
    print(f"scanned_files={scanned} findings={len(findings)} values_redacted=true scope=current_tracked_worktree")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
