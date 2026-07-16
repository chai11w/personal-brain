from __future__ import annotations

from dataclasses import dataclass

from .schema import BrainSchema


ARCHIVED_STATUS = "archived"


@dataclass(frozen=True)
class ArchiveMemoryResult:
    memory_id: int
    raw_message_id: int
    title: str | None
    content: str
    previous_status: str
    new_status: str
    embeddings_deleted: int


class MemoryOperations:
    """Small write operations for memory lifecycle management."""

    def __init__(self, schema: BrainSchema):
        self.schema = schema

    def archive_memory(self, memory_id: int) -> ArchiveMemoryResult:
        if memory_id <= 0:
            raise ValueError("memory_id must be positive")

        with self.schema.connect_write() as conn:
            row = conn.execute(
                """
                SELECT id, raw_message_id, title, content, status
                FROM memories
                WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {memory_id}")

            previous_status = str(row["status"])
            if previous_status != ARCHIVED_STATUS:
                conn.execute(
                    """
                    UPDATE memories
                    SET status = ?, updated_at = datetime('now', 'localtime')
                    WHERE id = ?
                    """,
                    (ARCHIVED_STATUS, memory_id),
                )
            cursor = conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            )

        return ArchiveMemoryResult(
            memory_id=memory_id,
            raw_message_id=int(row["raw_message_id"]),
            title=row["title"],
            content=row["content"],
            previous_status=previous_status,
            new_status=ARCHIVED_STATUS,
            embeddings_deleted=int(cursor.rowcount if cursor.rowcount is not None else 0),
        )
