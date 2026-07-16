from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RouterBuildResult:
    brain_index_path: Path
    topics_path: Path
    manifest_path: Path
    topic_count: int
    memory_count: int
    warnings: list[str]


class MemoryRouterBuilder:
    """Build lightweight routing indexes for Codex/AI memory access.

    The router does not perform semantic search or AI classification. It exposes
    a small navigation layer so AI callers can choose which evidence to load.
    """

    def __init__(self, database_path: Path, memory_dir: Path, brain_index_path: Path):
        self.database_path = database_path
        self.memory_dir = memory_dir
        self.brain_index_path = brain_index_path
        self.topics_path = memory_dir / "topics.json"
        self.manifest_path = memory_dir / "memory_manifest.json"

    def build(self) -> RouterBuildResult:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []

        with self._connect() as conn:
            if self._table_exists(conn, "memories") and self._has_column(conn, "memories", "extraction_run_id"):
                topics, manifest = self._build_from_ai_native_schema(conn)
            elif self._table_exists(conn, "memories"):
                topics, manifest = self._build_from_legacy_schema(conn, "memories")
                warnings.append(
                    "Database uses legacy memories table. Entries are routed as legacy_unprocessed, "
                    "not AI-extracted atomic memories."
                )
            else:
                topics = []
                manifest = []
                warnings.append("No memories table found. Router index is empty.")

            if self._table_exists(conn, "legacy_memories"):
                legacy_topics, legacy_manifest = self._build_from_legacy_schema(conn, "legacy_memories")
                topics = self._merge_topics(topics, legacy_topics)
                manifest.extend(legacy_manifest)
                warnings.append(
                    "legacy_memories exists. These rows are retained as evidence only, "
                    "not AI-extracted atomic memories."
                )

        self._write_json(self.topics_path, topics)
        self._write_json(self.manifest_path, manifest)
        brain_index = self._brain_index(topics, manifest, warnings)
        self._write_json(self.brain_index_path, brain_index)

        return RouterBuildResult(
            brain_index_path=self.brain_index_path,
            topics_path=self.topics_path,
            manifest_path=self.manifest_path,
            topic_count=len(topics),
            memory_count=len(manifest),
            warnings=warnings,
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=MEMORY")
        return conn

    def _build_from_ai_native_schema(self, conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        memory_topics = self._load_memory_topics(conn)
        topics = self._load_topics(conn, memory_topics)
        manifest = self._load_ai_memories(conn, memory_topics)
        return topics, manifest

    def _build_from_legacy_schema(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        rows = conn.execute(
            f"""
            SELECT id, content, source, sender, created_at, metadata
            FROM {table_name}
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
        manifest: list[dict[str, Any]] = []
        for row in rows:
            content = str(row["content"])
            manifest.append(
                {
                    "memory_id": f"legacy:{row['id']}",
                    "raw_message_id": None,
                    "title": self._shorten(content, 80),
                    "summary": self._shorten(content, 220),
                    "topics": ["legacy_unprocessed"],
                    "importance": None,
                    "confidence": None,
                    "memory_type": "legacy_raw_message",
                    "ai_generated": False,
                    "created_at": row["created_at"],
                    "evidence": {
                        "sqlite_path": self.database_path.as_posix(),
                        "sqlite_table": table_name,
                        "record_id": row["id"],
                    },
                }
            )

        topics = [
            {
                "topic_id": "legacy_unprocessed",
                "name": "Legacy unprocessed memories",
                "summary": (
                    "Records imported from the old prototype. They are raw or lightly stored text, "
                    "not AI-extracted atomic memories."
                ),
                "keywords": [],
                "memory_count": len(manifest),
                "manifest_refs": [item["memory_id"] for item in manifest[:50]],
                "ai_generated": False,
            }
        ] if manifest else []
        return topics, manifest

    @staticmethod
    def _merge_topics(
        primary: list[dict[str, Any]],
        extra: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {topic["topic_id"]: dict(topic) for topic in primary}
        for topic in extra:
            topic_id = topic["topic_id"]
            if topic_id not in by_id:
                by_id[topic_id] = dict(topic)
                continue
            existing = by_id[topic_id]
            existing["memory_count"] = int(existing.get("memory_count") or 0) + int(topic.get("memory_count") or 0)
            existing_refs = list(existing.get("manifest_refs") or [])
            for ref in topic.get("manifest_refs") or []:
                if ref not in existing_refs:
                    existing_refs.append(ref)
            existing["manifest_refs"] = existing_refs[:50]
        return list(by_id.values())

    def _load_memory_topics(self, conn: sqlite3.Connection) -> dict[int, list[dict[str, Any]]]:
        if not self._table_exists(conn, "memory_topics"):
            return {}
        rows = conn.execute(
            """
            SELECT mt.memory_id, mt.topic_id, mt.confidence, t.name
            FROM memory_topics mt
            JOIN memories m ON m.id = mt.memory_id
            LEFT JOIN topics t ON t.id = mt.topic_id
            WHERE m.status = 'active'
            """
        ).fetchall()
        by_memory: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            by_memory.setdefault(int(row["memory_id"]), []).append(
                {
                    "topic_id": row["topic_id"],
                    "name": row["name"] or str(row["topic_id"]),
                    "confidence": row["confidence"],
                }
            )
        return by_memory

    def _load_topics(
        self,
        conn: sqlite3.Connection,
        memory_topics: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        if not self._table_exists(conn, "topics"):
            return []
        counts: dict[int, int] = {}
        refs: dict[int, list[int]] = {}
        for memory_id, topic_links in memory_topics.items():
            for topic in topic_links:
                topic_id = int(topic["topic_id"])
                counts[topic_id] = counts.get(topic_id, 0) + 1
                refs.setdefault(topic_id, []).append(memory_id)

        rows = conn.execute("SELECT * FROM topics ORDER BY name ASC").fetchall()
        topics: list[dict[str, Any]] = []
        for row in rows:
            topic_id = int(row["id"])
            topics.append(
                {
                    "topic_id": topic_id,
                    "name": row["name"],
                    "summary": row["description"] if "description" in row.keys() else None,
                    "parent_topic_id": row["parent_topic_id"] if "parent_topic_id" in row.keys() else None,
                    "keywords": [],
                    "memory_count": counts.get(topic_id, 0),
                    "manifest_refs": refs.get(topic_id, [])[:50],
                    "ai_generated": True,
                }
            )
        return topics

    def _load_ai_memories(
        self,
        conn: sqlite3.Connection,
        memory_topics: dict[int, list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT *
            FROM memories
            WHERE status = 'active'
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
        manifest: list[dict[str, Any]] = []
        for row in rows:
            memory_id = int(row["id"])
            topic_links = memory_topics.get(memory_id, [])
            manifest.append(
                {
                    "memory_id": memory_id,
                    "raw_message_id": row["raw_message_id"],
                    "title": row["title"] if "title" in row.keys() else self._shorten(row["content"], 80),
                    "summary": self._shorten(row["content"], 260),
                    "memory_category": row["memory_category"] if "memory_category" in row.keys() else "未分类",
                    "topics": [topic["name"] for topic in topic_links],
                    "importance": row["importance"] if "importance" in row.keys() else None,
                    "confidence": row["confidence"] if "confidence" in row.keys() else None,
                    "memory_type": row["memory_type"] if "memory_type" in row.keys() else None,
                    "ai_generated": True,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"] if "updated_at" in row.keys() else row["created_at"],
                    "evidence": {
                        "sqlite_path": self.database_path.as_posix(),
                        "sqlite_table": "raw_messages",
                        "record_id": row["raw_message_id"],
                    },
                }
            )
        return manifest

    def _brain_index(
        self,
        topics: list[dict[str, Any]],
        manifest: list[dict[str, Any]],
        warnings: list[str],
    ) -> dict[str, Any]:
        return {
            "version": 1,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "product": "AI-native Personal Brain",
            "purpose": "Lightweight routing index for Codex and other AI callers.",
            "entrypoints": {
                "topics": self.topics_path.as_posix(),
                "memory_manifest": self.manifest_path.as_posix(),
                "sqlite": self.database_path.as_posix(),
            },
            "stats": {
                "topics": len(topics),
                "manifest_memories": len(manifest),
                "memory_categories": sorted(
                    {
                        str(item.get("memory_category"))
                        for item in manifest
                        if item.get("memory_category")
                    }
                ),
            },
            "routing_protocol": [
                "Read brain_index.json first.",
                "Read topics.json to identify likely topic areas.",
                "Read memory_manifest.json to select candidate memory ids.",
                "Read SQLite only for exact evidence or raw message recovery.",
                "Do not treat this router as RAG; embeddings and reranking are separate layers.",
            ],
            "architecture_constraints": [
                "Do not full-read SQLite unless explicitly requested.",
                "Do not infer facts without evidence records.",
                "Do not treat legacy_unprocessed entries as AI-extracted memories.",
            ],
            "warnings": warnings,
        }

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _shorten(text: str, limit: int) -> str:
        clean = " ".join(str(text).split())
        if len(clean) <= limit:
            return clean
        return clean[: limit - 1] + "..."

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        return row is not None

    @staticmethod
    def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row["name"] == column for row in rows)
