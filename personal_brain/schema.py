from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SchemaInitResult:
    database_path: Path
    schema_version: int
    migrated_legacy: bool
    tables: dict[str, int]
    warnings: list[str]


class BrainSchema:
    """Owns the AI-native database foundation.

    This module creates structure only. It does not implement ingestion,
    extraction, retrieval, or answering.
    """

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def initialize(self) -> SchemaInitResult:
        warnings: list[str] = []
        migrated_legacy = False
        with self.connect() as conn:
            migrated_legacy = self._move_legacy_memories(conn)
            if migrated_legacy:
                warnings.append(
                    "Moved old prototype memories table to legacy_memories. "
                    "Those rows are raw evidence, not AI atomic memories."
                )
            self._create_schema(conn)
            self._record_schema_version(conn)
            tables = self.table_counts(conn)
        return SchemaInitResult(
            database_path=self.database_path,
            schema_version=SCHEMA_VERSION,
            migrated_legacy=migrated_legacy,
            tables=tables,
            warnings=warnings,
        )

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            return self.table_counts(conn)

    def table_counts(self, conn: sqlite3.Connection) -> dict[str, int]:
        tables = [
            "raw_messages",
            "interaction_logs",
            "memory_extraction_runs",
            "memories",
            "memory_embeddings",
            "topics",
            "memory_topics",
            "entities",
            "memory_entities",
            "secure_items",
            "legacy_memories",
        ]
        counts: dict[str, int] = {}
        for table in tables:
            if self._table_exists(conn, table):
                row = conn.execute(f"SELECT COUNT(*) AS total FROM {table}").fetchone()
                counts[table] = int(row["total"])
        return counts

    def _move_legacy_memories(self, conn: sqlite3.Connection) -> bool:
        if not self._table_exists(conn, "memories"):
            return False
        if self._has_column(conn, "memories", "extraction_run_id"):
            return False
        if self._table_exists(conn, "legacy_memories"):
            return False
        conn.execute("ALTER TABLE memories RENAME TO legacy_memories")
        return True

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS raw_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                sender TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                metadata_json TEXT,
                processed_status TEXT NOT NULL DEFAULT 'pending',
                processed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_raw_messages_created_at
            ON raw_messages(created_at);

            CREATE INDEX IF NOT EXISTS idx_raw_messages_processed_status
            ON raw_messages(processed_status);

            CREATE TABLE IF NOT EXISTS interaction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                source TEXT NOT NULL,
                sender TEXT NOT NULL,
                user_text TEXT NOT NULL,
                mode TEXT NOT NULL,
                action TEXT NOT NULL,
                raw_message_id INTEGER,
                reply_text TEXT,
                evidence_json TEXT,
                status TEXT NOT NULL,
                error TEXT,
                latency_ms INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (raw_message_id) REFERENCES raw_messages(id)
            );

            CREATE INDEX IF NOT EXISTS idx_interaction_logs_created_at
            ON interaction_logs(created_at);

            CREATE INDEX IF NOT EXISTS idx_interaction_logs_source
            ON interaction_logs(source);

            CREATE TABLE IF NOT EXISTS memory_extraction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_message_id INTEGER NOT NULL,
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'succeeded',
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (raw_message_id) REFERENCES raw_messages(id)
            );

            CREATE INDEX IF NOT EXISTS idx_extraction_runs_raw_message_id
            ON memory_extraction_runs(raw_message_id);

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_message_id INTEGER NOT NULL,
                extraction_run_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                title TEXT,
                memory_category TEXT NOT NULL DEFAULT '未分类',
                memory_type TEXT NOT NULL,
                importance REAL NOT NULL CHECK (importance >= 0 AND importance <= 1),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (raw_message_id) REFERENCES raw_messages(id),
                FOREIGN KEY (extraction_run_id) REFERENCES memory_extraction_runs(id)
            );

            CREATE INDEX IF NOT EXISTS idx_memories_raw_message_id
            ON memories(raw_message_id);

            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(importance);

            CREATE INDEX IF NOT EXISTS idx_memories_updated_at
            ON memories(updated_at);

            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                PRIMARY KEY (memory_id, provider, model),
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );

            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                parent_topic_id INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS memory_topics (
                memory_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                reason TEXT,
                PRIMARY KEY (memory_id, topic_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                UNIQUE (name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                PRIMARY KEY (memory_id, entity_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id),
                FOREIGN KEY (entity_id) REFERENCES entities(id)
            );

            CREATE TABLE IF NOT EXISTS secure_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL UNIQUE,
                secret_type TEXT NOT NULL,
                username TEXT,
                encrypted_value TEXT NOT NULL,
                encryption_scheme TEXT NOT NULL,
                kdf_name TEXT NOT NULL,
                kdf_salt TEXT NOT NULL,
                kdf_iterations INTEGER NOT NULL,
                note TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_secure_items_label
            ON secure_items(label);
            """
        )
        if not self._has_column(conn, "memories", "memory_category"):
            conn.execute("ALTER TABLE memories ADD COLUMN memory_category TEXT NOT NULL DEFAULT '未分类'")
        conn.execute("UPDATE memories SET memory_category = '未分类' WHERE memory_category IS NULL OR memory_category = ''")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_category
            ON memories(memory_category)
            """
        )

    def _record_schema_version(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )

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
