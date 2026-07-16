from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .extractor import MEMORY_CATEGORIES, parse_json_object
from .llm import LLMClient
from .schema import BrainSchema


@dataclass(frozen=True)
class MemoryReviewResult:
    review_json: dict[str, Any]
    warning: str | None = None


class MemoryReviewer:
    """Dry-run memory quality review performed by the configured chat model."""

    def __init__(self, schema: BrainSchema, chat_model: LLMClient):
        self.schema = schema
        self.chat_model = chat_model

    def review(self, limit: int = 80, raw_message_id: int | None = None) -> MemoryReviewResult:
        if not self.chat_model.available:
            raise RuntimeError("chat model unavailable; configure chat_model before review")

        with self.schema.connect_readonly() as conn:
            memories = load_memories(conn, limit=limit, raw_message_id=raw_message_id)
            raw_messages = load_raw_messages(conn, memories)

        prompt = {
            "task": "review_personal_brain_memories_dry_run",
            "mode": "dry_run_only_do_not_modify_database",
            "stable_memory_categories": MEMORY_CATEGORIES,
            "review_goals": [
                "评估旧 memories 的标题、大类、内容、状态是否合理。",
                "识别低价值记忆、过细拆分、重复记忆、需要合并的记忆。",
                "保留 raw evidence，不建议删除原始 raw_messages。",
                "只给建议，不执行修改。",
                "优先让 Personal Brain 成为可信的第二大脑，而不是无限堆积的数据库。",
            ],
            "rules": [
                "所有建议必须使用中文，专有名词除外。",
                "每条 active memory 应该有清晰大类和动态主题。",
                "如果一条 raw_message 被拆成太多条，建议保留一条主记忆，其他标记 archive 或 merge_into。",
                "低价值记忆应建议 archive，而不是删除。",
                "不要编造 raw evidence 里没有的事实。",
                "如果当前状态已经合理，可以明确说 keep。",
                "recommendations 只输出需要改变的记忆，不要为每条 keep 记忆都生成建议。",
                "recommendations 最多输出 12 条。",
            ],
            "output_schema": {
                "summary": "总体质量评价",
                "overall_score": 0.0,
                "strengths": ["做得好的地方"],
                "problems": ["主要问题"],
                "recommendations": [
                    {
                        "memory_id": 1,
                        "action": "keep|update|merge_into|archive",
                        "target_memory_id": None,
                        "proposed_title": "建议标题",
                        "proposed_category": "建议大类",
                        "proposed_content": "建议内容",
                        "reason": "为什么这样处理",
                    }
                ],
                "merge_groups": [
                    {
                        "main_memory_id": 1,
                        "merged_memory_ids": [2, 3],
                        "reason": "为什么合并",
                    }
                ],
                "prompt_improvements": ["后续抽取提示词可怎么改"],
            },
            "raw_messages": raw_messages,
            "memories": memories,
        }
        answer = self.chat_model.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是 Personal Brain 的记忆质量复盘器。"
                        "你只输出 JSON，不执行修改，不写数据库。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        if not answer:
            raise RuntimeError("model returned empty memory review response")
        return MemoryReviewResult(review_json=parse_json_object(answer))


def load_memories(
    conn: sqlite3.Connection,
    limit: int,
    raw_message_id: int | None = None,
) -> list[dict[str, Any]]:
    where = ""
    params: list[Any] = []
    if raw_message_id is not None:
        where = "WHERE m.raw_message_id = ?"
        params.append(raw_message_id)
    params.append(limit)
    rows = conn.execute(
        f"""
        SELECT
            m.id, m.raw_message_id, m.title, m.content, m.memory_category,
            m.memory_type, m.importance, m.confidence, m.status, m.created_at,
            group_concat(t.name, ' | ') AS topics
        FROM memories m
        LEFT JOIN memory_topics mt ON mt.memory_id = m.id
        LEFT JOIN topics t ON t.id = mt.topic_id
        {where}
        GROUP BY m.id
        ORDER BY m.id ASC
        LIMIT ?
        """,
        params,
    ).fetchall()
    return [
        {
            "memory_id": int(row["id"]),
            "raw_message_id": int(row["raw_message_id"]),
            "title": row["title"],
            "content": row["content"],
            "memory_category": row["memory_category"],
            "memory_type": row["memory_type"],
            "importance": float(row["importance"]),
            "confidence": float(row["confidence"]),
            "status": row["status"],
            "topics": split_topics(row["topics"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def load_raw_messages(conn: sqlite3.Connection, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_ids = sorted({int(item["raw_message_id"]) for item in memories})
    if not raw_ids:
        return []
    placeholders = ",".join("?" for _ in raw_ids)
    rows = conn.execute(
        f"""
        SELECT id, content, source, sender, processed_status, created_at
        FROM raw_messages
        WHERE id IN ({placeholders})
        ORDER BY id ASC
        """,
        raw_ids,
    ).fetchall()
    return [
        {
            "raw_message_id": int(row["id"]),
            "content": row["content"],
            "source": row["source"],
            "sender": row["sender"],
            "processed_status": row["processed_status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def split_topics(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def format_memory_review(result: MemoryReviewResult) -> str:
    payload = result.review_json
    lines = [
        f"overall_score: {payload.get('overall_score')}",
        f"summary: {payload.get('summary')}",
    ]
    strengths = payload.get("strengths") or []
    if strengths:
        lines.append("")
        lines.append("strengths:")
        lines.extend(f"- {item}" for item in strengths)
    problems = payload.get("problems") or []
    if problems:
        lines.append("")
        lines.append("problems:")
        lines.extend(f"- {item}" for item in problems)
    recommendations = payload.get("recommendations") or []
    if recommendations:
        lines.append("")
        lines.append("recommendations:")
        for item in recommendations:
            memory_id = item.get("memory_id")
            action = item.get("action")
            reason = item.get("reason")
            title = item.get("proposed_title")
            category = item.get("proposed_category")
            target = item.get("target_memory_id")
            target_text = f" -> {target}" if target else ""
            lines.append(f"- #{memory_id} {action}{target_text}: {title} / {category} | {reason}")
    prompt_improvements = payload.get("prompt_improvements") or []
    if prompt_improvements:
        lines.append("")
        lines.append("prompt_improvements:")
        lines.extend(f"- {item}" for item in prompt_improvements)
    if result.warning:
        lines.extend(["", f"warning: {result.warning}"])
    return "\n".join(lines)
