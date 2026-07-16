from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import BrainSchema


@dataclass(frozen=True)
class MemorySummary:
    id: int
    title: str | None
    content: str
    memory_category: str
    memory_type: str
    importance: float
    confidence: float
    created_at: str
    topics: list[str]


@dataclass(frozen=True)
class MemoryDetail:
    summary: MemorySummary
    raw_message_id: int
    raw_content: str
    raw_source: str
    raw_sender: str
    raw_created_at: str
    extraction_run_id: int
    model_provider: str
    model_name: str
    prompt_version: str
    extraction_status: str
    entities: list[str]
    topic_reasons: list[str]


class MemoryView:
    """Read-only inspection view for AI-generated memories."""

    def __init__(self, schema: BrainSchema):
        self.schema = schema

    def list_memories(self, limit: int = 20) -> list[MemorySummary]:
        self.schema.initialize()
        with self.schema.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, title, content, memory_category, memory_type, importance, confidence, created_at
                FROM memories
                WHERE status = 'active'
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            summaries: list[MemorySummary] = []
            for row in rows:
                memory_id = int(row["id"])
                summaries.append(
                    MemorySummary(
                        id=memory_id,
                        title=row["title"],
                        content=row["content"],
                        memory_category=row["memory_category"],
                        memory_type=row["memory_type"],
                        importance=float(row["importance"]),
                        confidence=float(row["confidence"]),
                        created_at=row["created_at"],
                        topics=self._topics_for_memory(conn, memory_id),
                    )
                )
        return summaries

    def show_memory(self, memory_id: int) -> MemoryDetail:
        self.schema.initialize()
        with self.schema.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    m.id, m.title, m.content, m.memory_type, m.importance, m.confidence,
                    m.memory_category,
                    m.created_at, m.raw_message_id, m.extraction_run_id,
                    r.content AS raw_content, r.source AS raw_source, r.sender AS raw_sender,
                    r.created_at AS raw_created_at,
                    e.model_provider, e.model_name, e.prompt_version, e.status AS extraction_status
                FROM memories m
                JOIN raw_messages r ON r.id = m.raw_message_id
                JOIN memory_extraction_runs e ON e.id = m.extraction_run_id
                WHERE m.id = ?
                """,
                (memory_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"memory not found: {memory_id}")
            topics = self._topics_for_memory(conn, memory_id)
            topic_reasons = self._topic_reasons_for_memory(conn, memory_id)
            entities = self._entities_for_memory(conn, memory_id)

        summary = MemorySummary(
            id=int(row["id"]),
            title=row["title"],
            content=row["content"],
            memory_category=row["memory_category"],
            memory_type=row["memory_type"],
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            topics=topics,
        )
        return MemoryDetail(
            summary=summary,
            raw_message_id=int(row["raw_message_id"]),
            raw_content=row["raw_content"],
            raw_source=row["raw_source"],
            raw_sender=row["raw_sender"],
            raw_created_at=row["raw_created_at"],
            extraction_run_id=int(row["extraction_run_id"]),
            model_provider=row["model_provider"],
            model_name=row["model_name"],
            prompt_version=row["prompt_version"],
            extraction_status=row["extraction_status"],
            entities=entities,
            topic_reasons=topic_reasons,
        )

    def _topics_for_memory(self, conn, memory_id: int) -> list[str]:
        rows = conn.execute(
            """
            SELECT t.name
            FROM memory_topics mt
            JOIN topics t ON t.id = mt.topic_id
            WHERE mt.memory_id = ?
            ORDER BY mt.confidence DESC, t.name ASC
            """,
            (memory_id,),
        ).fetchall()
        return [row["name"] for row in rows]

    def _topic_reasons_for_memory(self, conn, memory_id: int) -> list[str]:
        rows = conn.execute(
            """
            SELECT t.name, mt.confidence, mt.reason
            FROM memory_topics mt
            JOIN topics t ON t.id = mt.topic_id
            WHERE mt.memory_id = ?
            ORDER BY mt.confidence DESC, t.name ASC
            """,
            (memory_id,),
        ).fetchall()
        reasons: list[str] = []
        for row in rows:
            reason = row["reason"] or ""
            reasons.append(f"{row['name']} ({float(row['confidence']):.2f}) {reason}".strip())
        return reasons

    def _entities_for_memory(self, conn, memory_id: int) -> list[str]:
        rows = conn.execute(
            """
            SELECT e.name, e.entity_type, me.confidence
            FROM memory_entities me
            JOIN entities e ON e.id = me.entity_id
            WHERE me.memory_id = ?
            ORDER BY me.confidence DESC, e.name ASC
            """,
            (memory_id,),
        ).fetchall()
        return [
            f"{row['name']}:{row['entity_type']} ({float(row['confidence']):.2f})"
            for row in rows
        ]


def format_memory_summary(memory: MemorySummary) -> str:
    title = memory.title or short_text(memory.content, 60)
    topics = ", ".join(memory.topics) if memory.topics else "no topics"
    return (
        f"#{memory.id} {title}\n"
        f"  category={memory.memory_category} type={memory.memory_type} importance={memory.importance:.2f} "
        f"confidence={memory.confidence:.2f} created={memory.created_at}\n"
        f"  topics={topics}\n"
        f"  content={short_text(memory.content, 120)}"
    )


def format_memory_detail(detail: MemoryDetail) -> str:
    lines = [
        f"memory #{detail.summary.id}",
        f"title: {detail.summary.title or '(none)'}",
        f"category: {detail.summary.memory_category}",
        f"type: {detail.summary.memory_type}",
        f"importance: {detail.summary.importance:.2f}",
        f"confidence: {detail.summary.confidence:.2f}",
        f"created_at: {detail.summary.created_at}",
        "",
        "AI memory:",
        detail.summary.content,
        "",
        f"topics: {', '.join(detail.summary.topics) if detail.summary.topics else '(none)'}",
    ]
    if detail.topic_reasons:
        lines.append("topic links:")
        lines.extend(f"- {reason}" for reason in detail.topic_reasons)
    lines.append(f"entities: {', '.join(detail.entities) if detail.entities else '(none)'}")
    lines.extend(
        [
            "",
            "raw evidence:",
            f"raw_message_id: {detail.raw_message_id}",
            f"source: {detail.raw_source}",
            f"sender: {detail.raw_sender}",
            f"created_at: {detail.raw_created_at}",
            detail.raw_content,
            "",
            "extraction run:",
            f"extraction_run_id: {detail.extraction_run_id}",
            f"model: {detail.model_provider}/{detail.model_name}",
            f"prompt_version: {detail.prompt_version}",
            f"status: {detail.extraction_status}",
        ]
    )
    return "\n".join(lines)


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."
