from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .answer import AnswerEngine, AnswerResult
from .config import BrainConfig, load_config
from .daily_report import DailyReportResult, DailyReportBuilder
from .extractor import IngestResult, MemoryExtractor
from .input_router import route_input
from .llm import EmbeddingClient, LLMClient
from .memory_ops import ArchiveMemoryResult, MemoryOperations
from .memory_view import MemoryDetail, MemorySummary, MemoryView
from .reviewer import MemoryReviewResult, MemoryReviewer
from .router import MemoryRouterBuilder, RouterBuildResult
from .schema import BrainSchema, SchemaInitResult
from .semantic import EmbedMemoriesResult, RecallResult, SemanticMemory
from .vault import SecureItemSecret, SecureItemSummary, SecureVault


class PersonalBrain:
    def __init__(self, config: BrainConfig | None = None):
        self.config = config or load_config()
        self.schema = BrainSchema(self.config.database_path)
        self.chat_model = LLMClient(self.config.chat_model)
        self.embedding_model = EmbeddingClient(self.config.embedding_model)
        self.vault = SecureVault(self.schema)
        self.memory_ops = MemoryOperations(self.schema)
        self.memory_view = MemoryView(self.schema)
        self.semantic_memory = SemanticMemory(
            schema=self.schema,
            embedding_client=self.embedding_model,
            embedding_config=self.config.embedding_model,
        )
        self.answer_engine = AnswerEngine(
            semantic_memory=self.semantic_memory,
            chat_model=self.chat_model,
        )

    @classmethod
    def from_config_file(cls, path: str | Path = "config.json") -> "PersonalBrain":
        return cls(load_config(path))

    def handle_message(self, text: str, sender: str = "me", source: str = "wechat") -> str:
        message = text.strip()
        if not message:
            return "我在。"
        try:
            result = self.ingest(message, source=source, sender=sender)
        except Exception as exc:
            return f"暂时没记住：{exc}"
        if result.memory_ids:
            if result.warning:
                return f"已记住，但后续处理有提醒：{result.warning}"
            return "已记住。"
        if result.warning:
            return f"已收到，但没有写入长期记忆：{result.warning}"
        return "已收到。"

    def init_db(self) -> SchemaInitResult:
        return self.schema.initialize()

    def ingest(
        self,
        text: str,
        source: str = "cli",
        sender: str = "me",
        rebuild_router: bool = True,
        source_message_id: str | None = None,
    ) -> IngestResult:
        input_route = route_input(text).as_debug_dict()
        extractor = MemoryExtractor(
            schema=self.schema,
            chat_model=self.chat_model,
            chat_config=self.config.chat_model,
        )
        result = extractor.ingest(
            text=text, source=source, sender=sender, source_message_id=source_message_id
        )
        recovered = bool(result.warning and result.warning.startswith("recovered committed"))
        warning = result.warning
        if result.memory_ids and not recovered:
            warning = self._embed_ingested_memories(result.memory_ids, warning)
        if rebuild_router and not recovered:
            self.build_router()
            return IngestResult(
                raw_message_id=result.raw_message_id,
                extraction_run_id=result.extraction_run_id,
                memory_ids=result.memory_ids,
                topic_ids=result.topic_ids,
                entity_ids=result.entity_ids,
                should_remember=result.should_remember,
                router_rebuilt=True,
                warning=warning,
                input_route=input_route,
            )
        if warning != result.warning:
            return IngestResult(
                raw_message_id=result.raw_message_id,
                extraction_run_id=result.extraction_run_id,
                memory_ids=result.memory_ids,
                topic_ids=result.topic_ids,
                entity_ids=result.entity_ids,
                should_remember=result.should_remember,
                router_rebuilt=result.router_rebuilt,
                warning=warning,
                input_route=input_route,
            )
        return IngestResult(
            raw_message_id=result.raw_message_id,
            extraction_run_id=result.extraction_run_id,
            memory_ids=result.memory_ids,
            topic_ids=result.topic_ids,
            entity_ids=result.entity_ids,
            should_remember=result.should_remember,
            router_rebuilt=result.router_rebuilt,
            warning=result.warning,
            input_route=input_route,
        )

    def _embed_ingested_memories(self, memory_ids: list[int], warning: str | None) -> str | None:
        if not self.config.embedding_model.enabled:
            return warning
        try:
            result = self.semantic_memory.embed_memories(memory_ids)
        except Exception as exc:
            return combine_warning(warning, f"embedding failed: {exc}")
        if result.warning:
            return combine_warning(warning, result.warning)
        return warning

    def build_router(self) -> RouterBuildResult:
        builder = MemoryRouterBuilder(
            database_path=self.config.database_path,
            memory_dir=self.config.memory_dir,
            brain_index_path=self.config.brain_index_path,
        )
        return builder.build()

    def test_chat(self, prompt: str) -> str:
        if not self.chat_model.available:
            return (
                f"chat model is not available. Enable chat_model and set "
                f"{self.config.chat_model.api_key_env}."
            )
        answer = self.chat_model.chat(
            [
                {
                    "role": "system",
                    "content": "你是 Personal Brain 的模型连通性测试助手。请简洁回答。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return answer or "empty response"

    def stats(self) -> str:
        counts = self.schema.stats()
        if not counts:
            return "database has no Personal Brain tables yet"
        return "\n".join(f"{table}: {count}" for table, count in counts.items())

    def secure_add(
        self,
        label: str,
        secret_type: str,
        secret: str,
        master_password: str,
        username: str | None = None,
        note: str | None = None,
    ) -> int:
        return self.vault.add_item(
            label=label,
            secret_type=secret_type,
            secret=secret,
            master_password=master_password,
            username=username,
            note=note,
        )

    def secure_list(self) -> list[SecureItemSummary]:
        return self.vault.list_items()

    def secure_get(self, label: str, master_password: str) -> SecureItemSecret:
        return self.vault.get_item(label, master_password)

    def memory_list(self, limit: int = 20) -> list[MemorySummary]:
        return self.memory_view.list_memories(limit=limit)

    def memory_show(self, memory_id: int) -> MemoryDetail:
        return self.memory_view.show_memory(memory_id)

    def archive_memory(self, memory_id: int, rebuild_router: bool = True) -> ArchiveMemoryResult:
        result = self.memory_ops.archive_memory(memory_id)
        if rebuild_router:
            self.build_router()
        return result

    def embed_missing_memories(self, limit: int = 100) -> EmbedMemoriesResult:
        return self.semantic_memory.embed_missing_memories(limit=limit)

    def recall(self, query: str, limit: int = 8) -> list[RecallResult]:
        return self.semantic_memory.recall(query=query, limit=limit)

    def ask(self, question: str, recall_limit: int = 8, evidence_limit: int = 5) -> AnswerResult:
        return self.answer_engine.ask(
            question=question,
            recall_limit=recall_limit,
            evidence_limit=evidence_limit,
        )

    def record_interaction(
        self,
        *,
        message_id: str | None,
        source: str,
        sender: str,
        user_text: str,
        mode: str,
        action: str,
        status: str,
        raw_message_id: int | None = None,
        reply_text: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        error: str | None = None,
        latency_ms: int | None = None,
        idempotency_key: str | None = None,
        processing_status: str = "pending",
        delivery_status: str = "unknown",
    ) -> int:
        evidence_json = json.dumps(evidence, ensure_ascii=False) if evidence is not None else None
        with self.schema.connect_write() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interaction_logs (
                    message_id, source, sender, user_text, mode, action,
                    raw_message_id, reply_text, evidence_json, status, error, latency_ms,
                    idempotency_key, processing_status, delivery_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    source,
                    sender,
                    user_text,
                    mode,
                    action,
                    raw_message_id,
                    reply_text,
                    evidence_json,
                    status,
                    error,
                    latency_ms,
                    idempotency_key,
                    processing_status,
                    delivery_status,
                ),
            )
            return int(cursor.lastrowid)

    def has_interaction_message(self, message_id: str | None) -> bool:
        if not message_id:
            return False
        with self.schema.connect_readonly() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM interaction_logs
                WHERE message_id = ?
                LIMIT 1
                """,
                (message_id,),
            ).fetchone()
            return row is not None

    def list_interactions(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.schema.connect_readonly() as conn:
            rows = conn.execute(
                """
                SELECT
                    id, message_id, source, sender, user_text, mode, action,
                    raw_message_id, reply_text, evidence_json, status, error,
                    latency_ms, created_at, idempotency_key, processing_status,
                    delivery_status, attempt_count, updated_at, delivered_at
                FROM interaction_logs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def claim_interaction(
        self, *, message_id: str, source: str, sender: str, user_text: str, mode: str,
        action: str = "pending", delivery_required: bool = True,
    ) -> tuple[int, bool]:
        """Atomically persist an inbox item before acknowledging its producer."""
        key = f"{source}:{message_id}"
        delivery = "pending" if delivery_required else "not_required"
        with self.schema.connect_write() as conn:
            cursor = conn.execute(
                """
                INSERT INTO interaction_logs (
                    message_id, source, sender, user_text, mode, action, status,
                    idempotency_key, processing_status, delivery_status
                ) VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?, 'pending', ?)
                ON CONFLICT(idempotency_key) WHERE idempotency_key IS NOT NULL DO NOTHING
                """,
                (message_id, source, sender, user_text, mode, action, key, delivery),
            )
            if cursor.rowcount:
                return int(cursor.lastrowid), True
            row = conn.execute(
                "SELECT id FROM interaction_logs WHERE idempotency_key=?", (key,)
            ).fetchone()
            return int(row["id"]), False

    def claim_interaction_processing(self, interaction_id: int) -> bool:
        with self.schema.connect_write() as conn:
            cursor = conn.execute(
                """
                UPDATE interaction_logs
                SET processing_status='processing', status='processing',
                    attempt_count=attempt_count+1, updated_at=datetime('now', 'localtime')
                WHERE id=? AND (
                    processing_status='pending'
                    OR (processing_status='processing' AND updated_at <= datetime('now', '-15 minutes'))
                )
                """,
                (interaction_id,),
            )
            return cursor.rowcount == 1

    def save_interaction_reply(
        self, interaction_id: int, *, action: str, reply_text: str,
        raw_message_id: int | None = None, evidence: list[dict[str, Any]] | None = None,
        processing_succeeded: bool = True, error: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        evidence_json = json.dumps(evidence, ensure_ascii=False) if evidence is not None else None
        processing = "ignored" if processing_succeeded and action == "ignored" else ("succeeded" if processing_succeeded else "failed")
        overall = "ignored" if processing == "ignored" else ("reply_pending" if processing_succeeded else "processing_failed")
        with self.schema.connect_write() as conn:
            conn.execute(
                """
                UPDATE interaction_logs
                SET action=?, reply_text=?, raw_message_id=?, evidence_json=?, error=?, latency_ms=?,
                    processing_status=?, delivery_status='pending', status=?,
                    updated_at=datetime('now', 'localtime')
                WHERE id=?
                """,
                (action, reply_text, raw_message_id, evidence_json, error, latency_ms,
                 processing, overall, interaction_id),
            )

    def mark_interaction_delivery(self, interaction_id: int, *, succeeded: bool, dry_run: bool = False, error: str | None = None) -> None:
        delivery = "dry_run" if dry_run else ("succeeded" if succeeded else "failed")
        with self.schema.connect_write() as conn:
            row = conn.execute(
                "SELECT processing_status FROM interaction_logs WHERE id=?", (interaction_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"interaction not found: {interaction_id}")
            if dry_run:
                overall = "dry_run"
            elif not succeeded:
                overall = "delivery_failed"
            else:
                overall = (
                    "succeeded" if row["processing_status"] == "succeeded"
                    else ("ignored" if row["processing_status"] == "ignored" else "processing_failed")
                )
            conn.execute(
                """
                UPDATE interaction_logs SET delivery_status=?, status=?, error=COALESCE(?, error),
                    delivered_at=CASE WHEN ? THEN datetime('now', 'localtime') ELSE delivered_at END,
                    updated_at=datetime('now', 'localtime') WHERE id=?
                """,
                (delivery, overall, error, succeeded and not dry_run, interaction_id),
            )

    def mark_interaction_ignored(self, interaction_id: int, *, action: str, note: str) -> None:
        with self.schema.connect_write() as conn:
            conn.execute(
                """
                UPDATE interaction_logs SET action=?, reply_text=?, processing_status='ignored',
                    delivery_status='not_required', status='ignored',
                    updated_at=datetime('now', 'localtime') WHERE id=?
                """,
                (action, note, interaction_id),
            )

    def get_interaction(self, interaction_id: int) -> dict[str, Any]:
        with self.schema.connect_readonly() as conn:
            row = conn.execute("SELECT * FROM interaction_logs WHERE id=?", (interaction_id,)).fetchone()
            if row is None:
                raise KeyError(f"interaction not found: {interaction_id}")
            return dict(row)

    def prepare_interaction_retry(self, interaction_id: int) -> dict[str, Any]:
        """CAS a saved reply for delivery by the bridge recovery worker."""
        with self.schema.connect_write() as conn:
            cursor = conn.execute(
                """
                UPDATE interaction_logs SET delivery_status='pending', status='reply_pending',
                    updated_at=datetime('now', 'localtime')
                WHERE id=? AND delivery_status IN ('pending', 'failed') AND reply_text IS NOT NULL
                """,
                (interaction_id,),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"interaction {interaction_id} is not reply_pending/delivery_failed with a saved reply"
                )
            row = conn.execute("SELECT * FROM interaction_logs WHERE id=?", (interaction_id,)).fetchone()
            return dict(row)

    def recoverable_interactions(self) -> list[dict[str, Any]]:
        with self.schema.connect_readonly() as conn:
            rows = conn.execute(
                """
                SELECT * FROM interaction_logs
                WHERE idempotency_key IS NOT NULL AND (processing_status='pending'
                   OR (processing_status='processing' AND updated_at <= datetime('now', '-15 minutes'))
                   OR delivery_status IN ('pending', 'failed'))
                ORDER BY id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def raw_reprocess(self, raw_message_id: int, *, recover_stale_minutes: int | None = None) -> IngestResult:
        extractor = MemoryExtractor(self.schema, self.chat_model, self.config.chat_model)
        result = extractor.reprocess(raw_message_id, recover_stale_minutes=recover_stale_minutes)
        if result.memory_ids:
            self._embed_ingested_memories(result.memory_ids, result.warning)
        return result

    def review_memories(self, limit: int = 80, raw_message_id: int | None = None) -> MemoryReviewResult:
        reviewer = MemoryReviewer(schema=self.schema, chat_model=self.chat_model)
        return reviewer.review(limit=limit, raw_message_id=raw_message_id)

    def daily_report(self, report_date: date, output_dir: Path) -> DailyReportResult:
        builder = DailyReportBuilder(schema=self.schema)
        return builder.build(report_date=report_date, output_dir=output_dir)

    def recent_report(self, hours: int, output_dir: Path) -> DailyReportResult:
        builder = DailyReportBuilder(schema=self.schema)
        return builder.build_recent_hours(hours=hours, output_dir=output_dir)


def combine_warning(first: str | None, second: str | None) -> str | None:
    if first and second:
        return f"{first}; {second}"
    return first or second
