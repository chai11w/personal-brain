from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import EmbeddingModelConfig
from .llm import EmbeddingClient
from .schema import BrainSchema


@dataclass(frozen=True)
class EmbedMemoriesResult:
    embedded_count: int
    skipped_count: int
    warning: str | None = None


@dataclass(frozen=True)
class RecallResult:
    memory_id: int
    score: float
    semantic_score: float
    exact_match_boost: float
    same_day_todo_boost: float
    todo_lifecycle_adjustment: float
    title: str | None
    content: str
    memory_category: str
    memory_type: str
    importance: float
    confidence: float
    created_at: str
    raw_message_id: int
    raw_content: str
    topics: list[str]


class SemanticMemory:
    """SQLite-backed embedding and recall layer for V0.1."""

    def __init__(
        self,
        schema: BrainSchema,
        embedding_client: EmbeddingClient,
        embedding_config: EmbeddingModelConfig,
    ):
        self.schema = schema
        self.embedding_client = embedding_client
        self.embedding_config = embedding_config

    def embed_missing_memories(self, limit: int = 100) -> EmbedMemoriesResult:
        if not self.embedding_client.available:
            return EmbedMemoriesResult(
                embedded_count=0,
                skipped_count=0,
                warning=f"embedding model unavailable; set {self.embedding_config.api_key_env}",
            )

        with self.schema.connect_write() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.title, m.content, m.memory_category, m.memory_type
                FROM memories m
                LEFT JOIN memory_embeddings e
                  ON e.memory_id = m.id
                 AND e.provider = ?
                 AND e.model = ?
                WHERE m.status = 'active'
                  AND e.memory_id IS NULL
                ORDER BY m.created_at ASC, m.id ASC
                LIMIT ?
                """,
                (self.embedding_config.provider, self.embedding_config.model, limit),
            ).fetchall()
            if not rows:
                return EmbedMemoriesResult(embedded_count=0, skipped_count=0)

            embedded = self._embed_rows(conn, rows)

        return EmbedMemoriesResult(
            embedded_count=embedded,
            skipped_count=max(0, len(rows) - embedded),
        )

    def embed_memories(self, memory_ids: list[int]) -> EmbedMemoriesResult:
        ids = sorted({int(memory_id) for memory_id in memory_ids if int(memory_id) > 0})
        if not ids:
            return EmbedMemoriesResult(embedded_count=0, skipped_count=0)

        if not self.embedding_client.available:
            return EmbedMemoriesResult(
                embedded_count=0,
                skipped_count=0,
                warning=f"embedding model unavailable; set {self.embedding_config.api_key_env}",
            )

        placeholders = ",".join("?" for _ in ids)
        with self.schema.connect_write() as conn:
            rows = conn.execute(
                f"""
                SELECT m.id, m.title, m.content, m.memory_category, m.memory_type
                FROM memories m
                LEFT JOIN memory_embeddings e
                  ON e.memory_id = m.id
                 AND e.provider = ?
                 AND e.model = ?
                WHERE m.status = 'active'
                  AND e.memory_id IS NULL
                  AND m.id IN ({placeholders})
                ORDER BY m.created_at ASC, m.id ASC
                """,
                (self.embedding_config.provider, self.embedding_config.model, *ids),
            ).fetchall()
            if not rows:
                return EmbedMemoriesResult(embedded_count=0, skipped_count=0)

            embedded = self._embed_rows(conn, rows)

        return EmbedMemoriesResult(
            embedded_count=embedded,
            skipped_count=max(0, len(rows) - embedded),
        )

    def recall(self, query: str, limit: int = 8) -> list[RecallResult]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("recall query cannot be empty")
        if not self.embedding_client.available:
            raise RuntimeError(f"embedding model unavailable; set {self.embedding_config.api_key_env}")

        query_vector = self.embedding_client.embed(clean_query)
        if not query_vector:
            raise RuntimeError("embedding model returned empty query vector")

        with self.schema.connect_readonly() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.id, m.title, m.content, m.memory_category, m.memory_type, m.importance, m.confidence,
                    m.created_at, m.raw_message_id, r.content AS raw_content,
                    e.vector_json
                FROM memory_embeddings e
                JOIN memories m ON m.id = e.memory_id
                JOIN raw_messages r ON r.id = m.raw_message_id
                WHERE e.provider = ?
                  AND e.model = ?
                  AND m.status = 'active'
                """,
                (self.embedding_config.provider, self.embedding_config.model),
            ).fetchall()
            scored: list[tuple[float, float, float, float, float, sqlite3.Row]] = []
            for row in rows:
                vector = json.loads(row["vector_json"])
                if isinstance(vector, list):
                    semantic_score = cosine_similarity(query_vector, [float(value) for value in vector])
                    lexical_boost = exact_match_boost(
                        clean_query,
                        [
                            row["title"],
                            row["content"],
                            row["memory_category"],
                            row["memory_type"],
                            row["raw_content"],
                            ", ".join(self._topics_for_memory(conn, int(row["id"]))),
                        ],
                    )
                    task_boost = same_day_todo_boost(clean_query, row)
                    lifecycle_adjustment = todo_lifecycle_adjustment(clean_query, row)
                    score = max(0.0, min(1.0, semantic_score + lexical_boost + task_boost + lifecycle_adjustment))
                    scored.append((score, semantic_score, lexical_boost, task_boost, lifecycle_adjustment, row))
            scored.sort(key=lambda item: item[0], reverse=True)
            results = [
                RecallResult(
                    memory_id=int(row["id"]),
                    score=score,
                    semantic_score=semantic_score,
                    exact_match_boost=lexical_boost,
                    same_day_todo_boost=task_boost,
                    todo_lifecycle_adjustment=lifecycle_adjustment,
                    title=row["title"],
                    content=row["content"],
                    memory_category=row["memory_category"],
                    memory_type=row["memory_type"],
                    importance=float(row["importance"]),
                    confidence=float(row["confidence"]),
                    created_at=row["created_at"],
                    raw_message_id=int(row["raw_message_id"]),
                    raw_content=row["raw_content"],
                    topics=self._topics_for_memory(conn, int(row["id"])),
                )
                for score, semantic_score, lexical_boost, task_boost, lifecycle_adjustment, row in scored[:limit]
            ]
        return results

    def _embedding_text(self, conn: sqlite3.Connection, memory_id: int, row: sqlite3.Row) -> str:
        topics = ", ".join(self._topics_for_memory(conn, memory_id))
        entities = ", ".join(self._entities_for_memory(conn, memory_id))
        parts = [
            f"title: {row['title'] or ''}",
            f"content: {row['content']}",
            f"category: {row['memory_category']}",
            f"type: {row['memory_type']}",
            f"topics: {topics}",
            f"entities: {entities}",
        ]
        return "\n".join(parts)

    def _embed_rows(self, conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> int:
        texts = [self._embedding_text(conn, int(row["id"]), row) for row in rows]
        vectors = self.embedding_client.embed_many(texts)
        embedded = 0
        for row, vector in zip(rows, vectors):
            memory_id = int(row["id"])
            dimension = len(vector)
            if self.embedding_config.dimension is not None and dimension != self.embedding_config.dimension:
                raise ValueError(
                    f"embedding dimension mismatch for memory {memory_id}: "
                    f"expected {self.embedding_config.dimension}, got {dimension}"
                )
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                    memory_id, provider, model, vector_json, dimension
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    self.embedding_config.provider,
                    self.embedding_config.model,
                    json.dumps(vector, separators=(",", ":")),
                    dimension,
                ),
            )
            embedded += 1
        return embedded

    @staticmethod
    def _topics_for_memory(conn: sqlite3.Connection, memory_id: int) -> list[str]:
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

    @staticmethod
    def _entities_for_memory(conn: sqlite3.Connection, memory_id: int) -> list[str]:
        rows = conn.execute(
            """
            SELECT e.name, e.entity_type
            FROM memory_entities me
            JOIN entities e ON e.id = me.entity_id
            WHERE me.memory_id = ?
            ORDER BY me.confidence DESC, e.name ASC
            """,
            (memory_id,),
        ).fetchall()
        return [f"{row['name']}:{row['entity_type']}" for row in rows]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def exact_match_boost(query: str, fields: list[str | None]) -> float:
    tokens = query_tokens(query)
    if not tokens:
        return 0.0
    haystack = "\n".join(field or "" for field in fields).lower()
    matched = sum(1 for token in tokens if token in haystack)
    if matched == 0:
        return 0.0
    return min(0.12, 0.04 * matched)


def query_tokens(query: str) -> list[str]:
    lowered = query.lower()
    return [
        match.group(0)
        for match in re.finditer(r"[a-z0-9_+\-.]{2,}|[\u4e00-\u9fff]{2,}", lowered)
    ]


def same_day_todo_boost(query: str, row: sqlite3.Row) -> float:
    if not is_todo_query(query):
        return 0.0

    text = "\n".join(
        str(part or "")
        for part in (
            row["title"],
            row["content"],
            row["memory_category"],
            row["memory_type"],
            row["raw_content"],
        )
    )
    boost = 0.0
    if row["memory_category"] == "临时待办":
        boost += 0.24
    if any(term in text for term in ("待办", "要做", "计划", "下一步", "准备", "别忘", "验证", "联系", "面试")):
        boost += 0.08
    if "今天" in query and "今天" in text:
        boost += 0.12
    if "今天" in query and "明天" in text and is_created_yesterday(str(row["created_at"] or "")):
        boost += 0.16
    return min(boost, 0.40)


def todo_lifecycle_adjustment(query: str, row: sqlite3.Row) -> float:
    if not is_todo_query(query):
        return 0.0
    if row["memory_category"] != "临时待办":
        return 0.0

    age_days = memory_age_days(str(row["created_at"] or ""))
    if age_days is None or age_days <= 2:
        return 0.0

    text = "\n".join(
        str(part or "")
        for part in (
            row["title"],
            row["content"],
            row["raw_content"],
        )
    )
    date_bound = looks_date_bound_todo(text)
    if date_bound:
        if age_days > 14:
            return -0.30
        if age_days > 7:
            return -0.24
        return -0.16

    if age_days > 30:
        return -0.16
    if age_days > 14:
        return -0.10
    if age_days > 7:
        return -0.04
    return 0.0


def is_todo_query(query: str) -> bool:
    clean = str(query or "")
    return any(term in clean for term in ("今天", "今日", "当天", "待办", "要做什么", "做什么", "任务", "计划"))


def is_created_yesterday(created_at: str) -> bool:
    try:
        created_date = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return False
    return created_date == (datetime.now().date() - timedelta(days=1))


def memory_age_days(created_at: str) -> int | None:
    try:
        created_date = datetime.strptime(created_at[:19], "%Y-%m-%d %H:%M:%S").date()
    except ValueError:
        return None
    return (datetime.now().date() - created_date).days


def looks_date_bound_todo(text: str) -> bool:
    clean = str(text or "")
    return any(
        term in clean
        for term in (
            "今天",
            "明天",
            "后天",
            "昨天",
            "本周",
            "这周",
            "下周",
            "周一",
            "周二",
            "周三",
            "周四",
            "周五",
            "周六",
            "周日",
            "星期",
            "上午",
            "下午",
            "晚上",
            "点前",
            "号",
            "月",
        )
    )


def format_recall_result(result: RecallResult) -> str:
    title = result.title or short_text(result.content, 60)
    topics = ", ".join(result.topics) if result.topics else "no topics"
    return (
        f"#{result.memory_id} score={result.score:.4f} "
        f"(semantic={result.semantic_score:.4f} exact={result.exact_match_boost:.4f} "
        f"todo={result.same_day_todo_boost:.4f} lifecycle={result.todo_lifecycle_adjustment:.4f}) {title}\n"
        f"  category={result.memory_category} type={result.memory_type} importance={result.importance:.2f} "
        f"confidence={result.confidence:.2f} created={result.created_at}\n"
        f"  topics={topics}\n"
        f"  memory={short_text(result.content, 180)}\n"
        f"  evidence raw_message_id={result.raw_message_id}: {short_text(result.raw_content, 180)}"
    )


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."
