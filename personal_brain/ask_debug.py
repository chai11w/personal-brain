from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .answer import AnswerResult
from .config import BrainConfig
from .semantic import RecallResult


@dataclass(frozen=True)
class AskDebugReportResult:
    output_path: Path


def write_ask_debug_report(
    *,
    result: AnswerResult,
    config: BrainConfig,
    recall_limit: int,
    evidence_limit: int,
    output_dir: Path,
) -> AskDebugReportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = f"ask-debug-{now:%Y-%m-%d-%H%M%S}-{slugify(result.question)}.md"
    output_path = output_dir / filename
    output_path.write_text(
        format_ask_debug_report(
            result=result,
            config=config,
            recall_limit=recall_limit,
            evidence_limit=evidence_limit,
            generated_at=now,
        ),
        encoding="utf-8",
    )
    return AskDebugReportResult(output_path=output_path)


def format_ask_debug_report(
    *,
    result: AnswerResult,
    config: BrainConfig,
    recall_limit: int,
    evidence_limit: int,
    generated_at: datetime,
) -> str:
    recalled = result.recalled or []
    used_ids = {item.memory_id for item in result.evidence}
    lines: list[str] = [
        "# Ask Debug Report",
        "",
        "## Metadata",
        "",
        f"- generated_at: {generated_at:%Y-%m-%d %H:%M:%S}",
        f"- question: {result.question}",
        f"- recall_limit: {recall_limit}",
        f"- evidence_limit: {evidence_limit}",
        f"- chat_model: {config.chat_model.provider}/{config.chat_model.model}",
        f"- embedding_model: {config.embedding_model.provider}/{config.embedding_model.model}",
        f"- answer_warning: {result.warning or 'none'}",
        "",
        "## Final Answer",
        "",
        result.answer.strip() or "(empty answer)",
        "",
        "## Recall Candidates",
        "",
        "| rank | used | memory | raw | category | final | semantic | exact | todo | lifecycle | title | topics |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if recalled:
        for rank, item in enumerate(recalled, start=1):
            lines.append(format_recall_row(rank, item, item.memory_id in used_ids))
    else:
        lines.append("| - | - | - | - | - | - | - | - | - | no recalled memories | - |")

    lines.extend(
        [
            "",
            "## Rerank Selected Evidence",
            "",
            "| rank | memory | raw | relevance | reason |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    if result.evidence:
        for rank, item in enumerate(result.evidence, start=1):
            lines.append(
                "| {rank} | {memory_id} | {raw_id} | {relevance:.4f} | {reason} |".format(
                    rank=rank,
                    memory_id=item.memory_id,
                    raw_id=item.recall.raw_message_id,
                    relevance=item.relevance,
                    reason=md_table_cell(item.reason or "(no reason)", 160),
                )
            )
    else:
        lines.append("| - | - | - | - | no evidence selected |")

    unused = [item for item in recalled if item.memory_id not in used_ids]
    lines.extend(
        [
            "",
            "## Recalled But Not Used",
            "",
            "| rank | memory | raw | final | category | title |",
            "|---:|---:|---:|---:|---|---|",
        ]
    )
    if unused:
        for rank, item in enumerate(unused, start=1):
            lines.append(
                "| {rank} | {memory_id} | {raw_id} | {score:.4f} | {category} | {title} |".format(
                    rank=rank,
                    memory_id=item.memory_id,
                    raw_id=item.raw_message_id,
                    score=item.score,
                    category=md_table_cell(item.memory_category, 60),
                    title=md_table_cell(item.title or short_text(item.content, 80), 120),
                )
            )
    else:
        lines.append("| - | - | - | - | - | all recalled candidates were used or none recalled |")

    lines.extend(
        [
            "",
            "## Review Checklist",
            "",
            "- Recall是否正确：",
            "- 是否有明显重复证据：",
            "- 是否有旧临时待办干扰：",
            "- evidence_limit是否过少或过多：",
            "- Rerank是否丢掉关键证据：",
            "- Citation是否能对应到依据：",
            "",
        ]
    )
    return "\n".join(lines)


def format_recall_row(rank: int, item: RecallResult, used: bool) -> str:
    return (
        "| {rank} | {used} | {memory_id} | {raw_id} | {category} | {score:.4f} | "
        "{semantic:.4f} | {exact:.4f} | {todo:.4f} | {lifecycle:.4f} | {title} | {topics} |"
    ).format(
        rank=rank,
        used="yes" if used else "no",
        memory_id=item.memory_id,
        raw_id=item.raw_message_id,
        category=md_table_cell(item.memory_category, 60),
        score=item.score,
        semantic=item.semantic_score,
        exact=item.exact_match_boost,
        todo=item.same_day_todo_boost,
        lifecycle=item.todo_lifecycle_adjustment,
        title=md_table_cell(item.title or short_text(item.content, 80), 120),
        topics=md_table_cell(", ".join(item.topics) if item.topics else "", 120),
    )


def slugify(text: str) -> str:
    clean = re.sub(r"\s+", "-", text.strip().lower())
    clean = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", clean)
    clean = clean.strip("-")
    if not clean:
        return "question"
    return clean[:40]


def md_table_cell(text: str, limit: int) -> str:
    clean = short_text(text, limit)
    return clean.replace("|", "\\|").replace("\n", " ")


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."
