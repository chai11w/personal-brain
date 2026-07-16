#!/usr/bin/env python3
"""Restore a SQLite backup into an isolated directory and verify it read-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from urllib.parse import quote

try:
    from .private_output import ensure_private_output
except ImportError:
    from private_output import ensure_private_output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {name: int(conn.execute(f'SELECT COUNT(*) FROM "{name.replace(chr(34), chr(34) * 2)}"').fetchone()[0]) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("backup", type=Path)
    parser.add_argument("restore_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    backup = args.backup.resolve()
    restore_dir = args.restore_dir.resolve()
    if not backup.is_file():
        raise SystemExit(f"backup does not exist: {backup}")
    restored = restore_dir / f"restored-{backup.name}"
    result_path = restored.with_suffix(restored.suffix + ".verification.json")
    ensure_private_output(restored)
    ensure_private_output(result_path)
    restore_dir.mkdir(parents=True, exist_ok=True)
    if restored.exists():
        raise SystemExit(f"refusing to overwrite restore target: {restored}")
    shutil.copy2(backup, restored)

    digest = sha256_file(restored)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    hash_matches = manifest is None or digest == manifest.get("sha256")
    uri = f"file:{quote(restored.as_posix(), safe='/:')}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_key_violations = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        table_counts = counts(conn)
    counts_match = manifest is None or table_counts == manifest.get("table_counts")
    result = {
        "restored_name": restored.name,
        "sha256": digest,
        "hash_matches_manifest": hash_matches,
        "integrity_check": integrity,
        "foreign_key_violations": foreign_key_violations,
        "table_counts_match_manifest": counts_match,
        "table_counts": table_counts,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"restored": str(restored), "verification": str(result_path), "ok": integrity == "ok" and hash_matches and counts_match and foreign_key_violations == 0}))
    return 0 if integrity == "ok" and hash_matches and counts_match and foreign_key_violations == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
