#!/usr/bin/env python3
"""Create a consistent private SQLite backup and a content-free manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
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


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    return {
        name: int(conn.execute(f'SELECT COUNT(*) FROM "{name.replace(chr(34), chr(34) * 2)}"').fetchone()[0])
        for name in names
    }


def inspect_database(conn: sqlite3.Connection) -> dict[str, object]:
    return {
        "integrity_check": conn.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_violations": len(conn.execute("PRAGMA foreign_key_check").fetchall()),
        "table_counts": table_counts(conn),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination_dir", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination_dir = args.destination_dir.resolve()
    if not source.is_file():
        raise SystemExit(f"source database does not exist: {source}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = destination_dir / f"{source.stem}-{stamp}.sqlite3"
    manifest_path = backup.with_suffix(backup.suffix + ".manifest.json")
    ensure_private_output(backup)
    ensure_private_output(manifest_path)
    destination_dir.mkdir(parents=True, exist_ok=True)
    if backup.exists():
        raise SystemExit(f"refusing to overwrite existing backup: {backup}")

    source_uri = f"file:{quote(source.as_posix(), safe='/:')}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)

    with sqlite3.connect(f"file:{quote(backup.as_posix(), safe='/:')}?mode=ro", uri=True) as check:
        check.row_factory = sqlite3.Row
        inspection = inspect_database(check)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "python_sqlite3_backup_api",
        "source_name": source.name,
        "backup_name": backup.name,
        "size_bytes": backup.stat().st_size,
        "sha256": sha256_file(backup),
        **inspection,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"backup": str(backup), "manifest": str(manifest_path), "integrity_check": inspection["integrity_check"]}))
    return 0 if inspection["integrity_check"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
