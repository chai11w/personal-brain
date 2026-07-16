from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .schema import BrainSchema


@dataclass(frozen=True)
class DailyReportResult:
    report_date: date
    start_at: str
    end_at: str
    output_path: Path
    markdown: str
    counts: dict[str, int]


class DailyReportBuilder:
    """Builds a local daily extraction report."""

    def __init__(self, schema: BrainSchema):
        self.schema = schema

    def build(self, report_date: date, output_dir: Path) -> DailyReportResult:
        start_at = datetime.combine(report_date, datetime.min.time())
        end_at = start_at + timedelta(days=1)
        output_path = output_dir / f"{report_date.isoformat()}.md"
        return self.build_window(
            start_at=start_at,
            end_at=end_at,
            output_path=output_path,
            title=f"Personal Brain 每日提取记录 {report_date.isoformat()}",
            window_label="自然日",
        )

    def build_recent_hours(self, hours: int, output_dir: Path, now: datetime | None = None) -> DailyReportResult:
        if hours <= 0:
            raise ValueError("hours must be positive")
        current = now or datetime.now()
        start_at = current - timedelta(hours=hours)
        output_path = output_dir / f"last-{hours}h-{current.strftime('%Y-%m-%d-%H%M')}.md"
        return self.build_window(
            start_at=start_at,
            end_at=current,
            output_path=output_path,
            title=f"Personal Brain 最近 {hours} 小时提取记录",
            window_label=f"最近 {hours} 小时",
        )

    def build_window(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        output_path: Path,
        title: str,
        window_label: str,
    ) -> DailyReportResult:
        with self.schema.connect_readonly() as conn:
            data = load_window_data(conn, start_at=start_at, end_at=end_at)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown = format_daily_report(
            title=title,
            window_label=window_label,
            start_at=start_at,
            end_at=end_at,
            data=data,
        )
        output_path.write_text(markdown + "\n", encoding="utf-8")
        return DailyReportResult(
            report_date=end_at.date(),
            start_at=format_dt(start_at),
            end_at=format_dt(end_at),
            output_path=output_path,
            markdown=markdown,
            counts=report_counts(data),
        )


def parse_report_date(value: str | None) -> date:
    if not value or value == "today":
        return date.today()
    if value == "yesterday":
        return date.today() - timedelta(days=1)
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_daily_data(conn: sqlite3.Connection, report_date: date) -> dict[str, list[dict[str, Any]]]:
    start_at = datetime.combine(report_date, datetime.min.time())
    end_at = start_at + timedelta(days=1)
    return load_window_data(conn, start_at=start_at, end_at=end_at)


def load_window_data(conn: sqlite3.Connection, start_at: datetime, end_at: datetime) -> dict[str, list[dict[str, Any]]]:
    start = format_dt(start_at)
    end = format_dt(end_at)
    return {
        "raw_messages": fetch_rows(
            conn,
            """
            SELECT id, created_at, processed_at, source, sender, processed_status,
                   content, metadata_json
            FROM raw_messages
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at, id
            """,
            (start, end),
        ),
        "extraction_runs": fetch_rows(
            conn,
            """
            SELECT id, raw_message_id, created_at, model_provider, model_name,
                   prompt_version, status, error, output_json
            FROM memory_extraction_runs
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at, id
            """,
            (start, end),
        ),
        "memories": fetch_rows(
            conn,
            """
            SELECT id, raw_message_id, extraction_run_id, created_at, updated_at,
                   title, memory_category, memory_type, importance, confidence,
                   status, content
            FROM memories
            WHERE (created_at >= ? AND created_at < ?)
               OR (updated_at >= ? AND updated_at < ?)
            ORDER BY created_at, id
            """,
            (start, end, start, end),
        ),
        "interactions": fetch_rows(
            conn,
            """
            SELECT id, message_id, created_at, source, sender, mode, action,
                   status, raw_message_id, latency_ms, user_text, reply_text,
                   evidence_json, error
            FROM interaction_logs
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at, id
            """,
            (start, end),
        ),
    }


def fetch_rows(conn: sqlite3.Connection, query: str, params: tuple[str, ...]) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def report_counts(data: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        "raw_messages": len(data["raw_messages"]),
        "extraction_runs": len(data["extraction_runs"]),
        "memories": len(data["memories"]),
        "interactions": len(data["interactions"]),
        "issue_markers": len(build_issue_markers(data)),
    }


def format_dt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def format_daily_report(
    *,
    title: str,
    window_label: str,
    start_at: datetime,
    end_at: datetime,
    data: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        f"# {title}",
        "",
        "用途：提取指定时间窗口内的记录，并用固定规则标记可能需要回看的链路问题；不调用 AI，不修改数据。",
        "提醒：未来 Codex 解读本报告前，应先阅读 .agents/project_memory.md。",
        "隐私：本报告可能包含用户原文，只应保留在本地。",
        "",
        "## 时间窗口",
        "",
        f"- window: {window_label}",
        f"- start_at: {format_dt(start_at)}",
        f"- end_at: {format_dt(end_at)}",
        "",
        "## 数量",
        "",
        f"- raw_messages: {len(data['raw_messages'])}",
        f"- extraction_runs: {len(data['extraction_runs'])}",
        f"- memories_created_or_updated: {len(data['memories'])}",
        f"- interactions: {len(data['interactions'])}",
        f"- issue_markers: {len(build_issue_markers(data))}",
    ]

    lines.extend(["", "## 链路问题标记", ""])
    append_issue_markers(lines, build_issue_markers(data))
    lines.extend(["", "## 原文 -> 实际存入的记忆", ""])
    append_raw_to_memories(lines, data["raw_messages"], data["memories"])
    lines.extend(["", "## 原文详情", ""])
    append_raw_messages(lines, data["raw_messages"])
    lines.extend(["", "## 提取详情", ""])
    append_extraction_runs(lines, data["extraction_runs"])
    lines.extend(["", "## 记忆详情", ""])
    append_memories(lines, data["memories"])
    lines.extend(["", "## 交互详情", ""])
    append_interactions(lines, data["interactions"])
    return "\n".join(lines).rstrip()


def short_excerpt(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def append_raw_to_memories(
    lines: list[str],
    raw_messages: list[dict[str, Any]],
    memories: list[dict[str, Any]],
) -> None:
    if not raw_messages:
        lines.append("- none")
        return
    memories_by_raw: dict[int, list[dict[str, Any]]] = {}
    for memory in memories:
        memories_by_raw.setdefault(int(memory["raw_message_id"]), []).append(memory)
    for raw in raw_messages:
        raw_id = int(raw["id"])
        stored = memories_by_raw.get(raw_id, [])
        lines.extend(
            [
                f"### raw_message {raw_id}",
                "",
                f"- created_at: {raw['created_at']}",
                f"- processed_status: {raw['processed_status']}",
                "",
                "原文：",
                "",
                fenced(raw["content"]),
                "",
                f"实际存入记忆：{len(stored)}",
                "",
            ]
        )
        if not stored:
            lines.append("- none")
            lines.append("")
            continue
        for memory in stored:
            lines.extend(
                [
                    f"#### memory {memory['id']}",
                    "",
                    f"- title: {memory['title'] or ''}",
                    f"- status: {memory['status']}",
                    f"- category: {memory['memory_category']}",
                    f"- type: {memory['memory_type']}",
                    f"- importance: {memory['importance']}",
                    f"- confidence: {memory['confidence']}",
                    "",
                    "存入内容：",
                    "",
                    fenced(memory["content"]),
                    "",
                ]
            )


def append_issue_markers(lines: list[str], markers: list[str]) -> None:
    if not markers:
        lines.append("- none")
        return
    lines.extend(f"- {marker}" for marker in markers)


def build_issue_markers(data: dict[str, list[dict[str, Any]]]) -> list[str]:
    markers: list[str] = []
    memories_by_raw: dict[int, list[dict[str, Any]]] = {}
    for memory in data["memories"]:
        memories_by_raw.setdefault(int(memory["raw_message_id"]), []).append(memory)

    raw_rows = data["raw_messages"]
    for index, raw in enumerate(raw_rows):
        raw_id = int(raw["id"])
        status = str(raw["processed_status"])
        content = str(raw["content"] or "")
        stored = memories_by_raw.get(raw_id, [])
        if status == "failed":
            markers.append(f"raw_message {raw_id}: 原文处理失败")
        if status == "ignored":
            later_rows = raw_rows[index + 1 :]
            if should_mark_ignored_raw(content, later_rows, memories_by_raw):
                markers.append(f"raw_message {raw_id}: 原文被忽略，未进入长期记忆")
        if user_explicitly_asked_to_remember(content) and not stored and should_mark_ignored_raw(content, raw_rows[index + 1 :], memories_by_raw):
            markers.append(f"raw_message {raw_id}: 用户明确要求记住，但没有实际存入记忆")

    for run in data["extraction_runs"]:
        run_id = int(run["id"])
        if run["status"] != "succeeded":
            markers.append(f"extraction_run {run_id}: 提取失败：{run['error'] or 'unknown error'}")
        else:
            summary = summarize_extraction_output(run["output_json"])
            if summary == "invalid JSON" or "missing" in summary:
                markers.append(f"extraction_run {run_id}: 模型输出结构异常：{summary}")

    seen_active: dict[str, int] = {}
    for memory in data["memories"]:
        memory_id = int(memory["id"])
        status = str(memory["status"])
        content = str(memory["content"] or "")
        if contains_markdown_noise(content):
            markers.append(f"memory {memory_id}: 入库内容含 Markdown 噪音")
        if status not in ("active", "archived"):
            markers.append(f"memory {memory_id}: 当前状态为 {status}")
        key = normalize_for_duplicate_check(content)
        if status == "active" and key:
            previous = seen_active.get(key)
            if previous:
                markers.append(f"memory {memory_id}: 疑似重复 active 记忆，接近 memory {previous}")
            else:
                seen_active[key] = memory_id

    for interaction in data["interactions"]:
        interaction_id = int(interaction["id"])
        reply = str(interaction["reply_text"] or "")
        if interaction["status"] != "succeeded":
            markers.append(f"interaction {interaction_id}: 交互失败：{interaction['error'] or 'unknown error'}")
        if contains_legacy_citation(reply):
            markers.append(f"interaction {interaction_id}: 回复仍使用旧证据格式")
        if contains_markdown_noise(reply):
            markers.append(f"interaction {interaction_id}: 回复含 Markdown 噪音")
    return markers


def user_explicitly_asked_to_remember(text: str) -> bool:
    return any(token in text for token in ("记得", "要记", "记住", "帮我记"))


def should_mark_ignored_raw(
    content: str,
    later_raw_rows: list[dict[str, Any]],
    memories_by_raw: dict[int, list[dict[str, Any]]],
) -> bool:
    if is_low_signal_ignored_raw(content):
        return False
    if is_covered_by_later_stored_raw(content, later_raw_rows, memories_by_raw):
        return False
    return True


def is_low_signal_ignored_raw(text: str) -> bool:
    clean = re.sub(r"\s+", "", str(text or ""))
    if not clean:
        return True
    if len(clean) <= 2 and re.fullmatch(r"[\d.。!！?？,，、]+", clean):
        return True
    return False


def is_covered_by_later_stored_raw(
    content: str,
    later_raw_rows: list[dict[str, Any]],
    memories_by_raw: dict[int, list[dict[str, Any]]],
) -> bool:
    current = normalize_raw_for_coverage(content)
    if len(current) < 8:
        return False
    for later in later_raw_rows[:8]:
        later_id = int(later["id"])
        if not memories_by_raw.get(later_id):
            continue
        later_content = normalize_raw_for_coverage(str(later["content"] or ""))
        if len(later_content) < 8:
            continue
        if current in later_content:
            return True
        longest = SequenceMatcher(None, current, later_content).find_longest_match(
            0,
            len(current),
            0,
            len(later_content),
        )
        if longest.size >= 12 and longest.size / max(1, len(current)) >= 0.62:
            return True
        similarity = SequenceMatcher(None, current, later_content).ratio()
        if similarity >= 0.86:
            return True
    return False


def normalize_raw_for_coverage(text: str) -> str:
    clean = str(text or "").lower()
    return "".join(char for char in clean if char.isalnum())


def contains_markdown_noise(text: str) -> bool:
    return "**" in text or "`" in text or bool(re.search(r"(?m)^\s{0,3}#{1,6}\s+", text))


def contains_legacy_citation(text: str) -> bool:
    return "memory_id=" in text or "raw_message_id=" in text


def normalize_for_duplicate_check(text: str) -> str:
    clean = re.sub(r"\s+", "", text.lower())
    clean = clean.replace("，", ",").replace("。", ".")
    return clean[:180]


def append_raw_messages(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        lines.append("- none")
        return
    for row in rows:
        lines.extend(
            [
                f"### raw_message {row['id']}",
                "",
                f"- created_at: {row['created_at']}",
                f"- processed_at: {row['processed_at'] or ''}",
                f"- source: {row['source']}",
                f"- sender: {row['sender']}",
                f"- processed_status: {row['processed_status']}",
                f"- metadata_json: {row['metadata_json'] or ''}",
                "",
                fenced(row["content"]),
                "",
            ]
        )


def append_extraction_runs(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        lines.append("- none")
        return
    for row in rows:
        lines.extend(
            [
                f"### extraction_run {row['id']} / raw_message {row['raw_message_id']}",
                "",
                f"- created_at: {row['created_at']}",
                f"- model: {row['model_provider']} / {row['model_name']}",
                f"- prompt_version: {row['prompt_version']}",
                f"- status: {row['status']}",
                f"- error: {row['error'] or ''}",
                f"- output_summary: {summarize_extraction_output(row['output_json'])}",
                "",
                "Output JSON:",
                "",
                fenced(row["output_json"]),
                "",
            ]
        )


def append_memories(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        lines.append("- none")
        return
    for row in rows:
        lines.extend(
            [
                f"### memory {row['id']} / raw_message {row['raw_message_id']}",
                "",
                f"- extraction_run_id: {row['extraction_run_id']}",
                f"- created_at: {row['created_at']}",
                f"- updated_at: {row['updated_at']}",
                f"- status: {row['status']}",
                f"- title: {row['title'] or ''}",
                f"- category: {row['memory_category']}",
                f"- type: {row['memory_type']}",
                f"- importance: {row['importance']}",
                f"- confidence: {row['confidence']}",
                "",
                fenced(row["content"]),
                "",
            ]
        )


def append_interactions(lines: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        lines.append("- none")
        return
    for row in rows:
        evidence = summarize_evidence(row["evidence_json"])
        lines.extend(
            [
                f"### interaction {row['id']}",
                "",
                f"- created_at: {row['created_at']}",
                f"- source: {row['source']}",
                f"- sender: {row['sender']}",
                f"- mode/action/status: {row['mode']} / {row['action']} / {row['status']}",
                f"- message_id: {row['message_id'] or ''}",
                f"- raw_message_id: {row['raw_message_id'] or ''}",
                f"- latency_ms: {row['latency_ms'] or ''}",
                f"- error: {row['error'] or ''}",
                f"- evidence: {evidence}",
                "",
                "User text:",
                "",
                fenced(row["user_text"]),
                "",
                "Reply text:",
                "",
                fenced(row["reply_text"] or ""),
                "",
                "Evidence JSON:",
                "",
                fenced(row["evidence_json"] or ""),
                "",
            ]
        )


def summarize_extraction_output(output_json: str) -> str:
    try:
        payload = json.loads(output_json)
    except json.JSONDecodeError:
        return "invalid JSON"
    memories = payload.get("atomic_memories")
    if not isinstance(memories, list):
        return "atomic_memories missing or not a list"
    titles = [str(item.get("title") or "untitled") for item in memories if isinstance(item, dict)]
    if not titles:
        return "0 memories"
    return f"{len(titles)} memories: " + "; ".join(titles)


def summarize_evidence(evidence_json: str | None) -> str:
    if not evidence_json:
        return ""
    try:
        evidence = json.loads(evidence_json)
    except json.JSONDecodeError:
        return "invalid JSON"
    if not isinstance(evidence, list):
        return "not a list"
    chunks = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        memory_id = item.get("memory_id")
        raw_message_id = item.get("raw_message_id")
        relevance = item.get("relevance")
        chunks.append(f"memory {memory_id}/raw {raw_message_id}/rel {relevance}")
    return "; ".join(chunks)


def fenced(text: str) -> str:
    fence = "```"
    if "```" in text:
        fence = "````"
    return f"{fence}\n{text}\n{fence}"
