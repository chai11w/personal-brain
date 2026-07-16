from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any

from .config import ChatModelConfig
from .llm import LLMClient
from .schema import BrainSchema


PROMPT_VERSION = "memory-extraction-v7"
DUPLICATE_SIMILARITY_THRESHOLD = 0.92


MEMORY_CATEGORIES = [
    "现有项目改进",
    "未来产品设想",
    "生活感悟",
    "产品使用技巧",
    "自身认知更新",
    "学习",
    "技术思考",
    "人际关系",
    "工作流方法",
    "信息安全",
    "临时待办",
    "其他",
]


@dataclass(frozen=True)
class IngestResult:
    raw_message_id: int
    extraction_run_id: int | None
    memory_ids: list[int]
    topic_ids: list[int]
    entity_ids: list[int]
    should_remember: bool
    router_rebuilt: bool
    warning: str | None = None
    input_route: dict[str, str] | None = None


class MemoryExtractor:
    """Turn casual user input into AI-generated atomic memories."""

    def __init__(
        self,
        schema: BrainSchema,
        chat_model: LLMClient,
        chat_config: ChatModelConfig,
    ):
        self.schema = schema
        self.chat_model = chat_model
        self.chat_config = chat_config

    def ingest(
        self,
        text: str,
        source: str = "cli",
        sender: str = "me",
        metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        content = text.strip()
        if not content:
            raise ValueError("ingest text cannot be empty")

        self.schema.initialize()
        raw_message_id = self._insert_raw_message(content, source, sender, metadata)

        if not self.chat_model.available:
            extraction_run_id = self._record_failed_run(
                raw_message_id,
                content,
                "chat model is not available",
            )
            self._mark_raw_status(raw_message_id, "failed")
            return IngestResult(
                raw_message_id=raw_message_id,
                extraction_run_id=extraction_run_id,
                memory_ids=[],
                topic_ids=[],
                entity_ids=[],
                should_remember=False,
                router_rebuilt=False,
                warning=f"chat model unavailable; set {self.chat_config.api_key_env}",
            )

        try:
            output_text = self._call_model(content)
            payload = parse_json_object(output_text)
        except Exception as exc:
            extraction_run_id = self._record_failed_run(raw_message_id, content, str(exc))
            self._mark_raw_status(raw_message_id, "failed")
            raise RuntimeError(f"memory extraction failed: {exc}") from exc

        payload = preserve_exact_technical_tokens(content, payload)
        payload = preserve_personal_brain_feedback(content, payload)
        payload = preserve_learning_note(content, payload)
        payload = preserve_temporary_todo(content, payload)
        return self._persist_extraction(raw_message_id, content, payload)

    def _call_model(self, content: str) -> str:
        system_prompt = (
            "你是 AI-native Personal Brain 的记忆提取器。"
            "你的任务不是聊天，而是把用户的随意输入整理成长期记忆。"
            "你可以去掉口语、重复和噪音，但不能改变用户原意。"
            "默认优先形成少量高密度记忆，而不是把一段完整想法切碎。"
            "只有当输入里包含彼此独立、后续会分别检索和更新的长期事实时，才拆成多条 atomic memories。"
            "使用规则、并列要点、同一愿景、同一项目决策、同一段反思，通常应合并成一条结构化记忆。"
            "当用户只是记录事实、缺点、观察或体验时，只忠实保存事实本身，不要主动添加建议、解决方案、规避动作或产品 coaching。"
            "所有标题、主题、说明、原因必须使用中文，除非是 ChatGPT、Codex、GitHub、API key 这类专有名词。"
            "只输出 JSON，不要输出 Markdown，不要解释。"
        )
        user_prompt = {
            "task": "extract_personal_memory",
            "stable_memory_categories": MEMORY_CATEGORIES,
            "rules": [
                "保留用户原意，不要替用户拔高成他没说过的结论。",
                "技术学习笔记中的代码标识符、函数名、文件名和扩展名必须按原文精确保留，包括大小写和前缀；例如 json.load、json.loads、json.dumps、memory.json 不得改成 .load、.loads、.dumps、memory. 或其他缺失形式。",
                "改写成简洁的第三人称长期记忆。",
                "宁可一条记忆内容稍完整，也不要把同一个 raw_message 机械拆成很多低密度记忆。",
                "当输入是编号列表、使用规则、测试说明或同一主题下的多个并列要点时，优先抽取为一条结构化记忆。",
                "只有当不同要点属于不同大类、不同时间计划、不同对象或后续需要独立作废/更新时，才拆分为多条记忆。",
                "每条 atomic memory 应表达一个完整可复用判断；不要生成只改写半句话的低价值记忆。",
                "不要把用户没有说过的建议写进记忆。例如用户只说某软件画线会被图片遮挡，就只记录这个缺陷，不要补充“使用时应避免画长直线”。",
                "每条 atomic memory 必须选择一个 stable_memory_categories 中的大类。",
                "topics 仍然由 AI 动态生成，但必须优先复用语义相近的中文主题名，不要为每条记忆凭空新造一个主题。",
                "topic 是小方向，memory_category 是大方向；不要把二者混在一起。",
                "只记 durable 的偏好、决定、想法、原则、计划、反思、自我认知、处事方式或产品方向。",
                "临时命令、普通提问、能力询问、寒暄、过短且无上下文的吐槽，应 set should_remember=false。",
                "明确的短期待办、提醒、会议/出行准备、带有时间边界的别忘事项，应 should_remember=true，并归入“临时待办”；不要把它们拔高成长期原则。",
                "如果用户围绕 Personal Brain、个人记忆系统、质量报告、消息入口、去重、检索、RAG 或 embedding 提出改进、缺陷、近期修复或未来方向，即使句式是问题，也应作为长期项目反馈记住。",
                "判断问题式输入时，区分‘询问当前能力’和‘提出产品方向’：例如‘能不能让质量报告单独分类缺陷、修复和未来方向’属于项目改进，应 should_remember=true。",
                "如果用户是在记录系统改进方向，要优先归入“现有项目改进”。",
                "如果用户是在描述未来产品形态、第二个我、数字分身、接入其他软件，优先归入“未来产品设想”。",
                "如果用户是在记录短概念、定义、区别、类比、术语理解，或类似“X 就是 Y”“X 指的是 Y”“X 可以理解为 Y”的学习笔记，且主要价值是以后理解/复习，应 should_remember=true，并归入“学习”。",
                "例如“memory+recall就是储存加调取的组合”“长期记忆就是未来大概率会重复利用的信息”“harness 类似于测评”这类紧凑概念笔记，应保存为“学习”。",
                "不要把概念学习误归入“自身认知更新”；只有描述用户性格、学习方式、情绪模式、个人原则时，才归入“自身认知更新”。",
                "不要把普通概念学习误归入“技术思考”；只有技术判断、架构取舍、实现策略或是否改造某技术方案的推理，才归入“技术思考”。",
                "如果用户只是在说某个工具名字，不要生成“用户知道某工具”这种低价值记忆。",
            ],
            "output_schema": {
                "should_remember": True,
                "reason": "why this should or should not be remembered",
                "atomic_memories": [
                    {
                        "title": "short title",
                        "content": "AI-rewritten atomic memory",
                        "memory_category": "one stable category from stable_memory_categories",
                        "memory_type": "preference|principle|decision|idea|plan|reflection|fact|other",
                        "importance": 0.0,
                        "confidence": 0.0,
                        "topics": [
                            {
                                "name": "dynamic topic name",
                                "description": "what this topic means",
                                "confidence": 0.0,
                                "reason": "why linked",
                            }
                        ],
                        "entities": [
                            {
                                "name": "entity name",
                                "type": "person|product|tool|project|concept|other",
                                "description": "optional short description",
                                "confidence": 0.0,
                            }
                        ],
                    }
                ],
            },
            "user_input": content,
        }
        answer = self.chat_model.chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_prompt, ensure_ascii=False),
                },
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        if not answer:
            raise RuntimeError("model returned empty response")
        return answer

    def _insert_raw_message(
        self,
        content: str,
        source: str,
        sender: str,
        metadata: dict[str, Any] | None,
    ) -> int:
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        with self.schema.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO raw_messages (content, source, sender, metadata_json, processed_status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (content, source, sender, metadata_json),
            )
            return int(cursor.lastrowid)

    def _record_failed_run(self, raw_message_id: int, content: str, error: str) -> int:
        with self.schema.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memory_extraction_runs (
                    raw_message_id, model_provider, model_name, prompt_version,
                    input_hash, output_json, status, error
                )
                VALUES (?, ?, ?, ?, ?, ?, 'failed', ?)
                """,
                (
                    raw_message_id,
                    self.chat_config.provider,
                    self.chat_config.model,
                    PROMPT_VERSION,
                    input_hash(content),
                    "{}",
                    error,
                ),
            )
            return int(cursor.lastrowid)

    def _persist_extraction(
        self,
        raw_message_id: int,
        raw_content: str,
        payload: dict[str, Any],
    ) -> IngestResult:
        should_remember = bool(payload.get("should_remember", False))
        memories = payload.get("atomic_memories") or []
        if not isinstance(memories, list):
            raise ValueError("atomic_memories must be a list")

        output_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        memory_ids: list[int] = []
        topic_ids: list[int] = []
        entity_ids: list[int] = []
        skipped_duplicates: list[tuple[str, int]] = []

        with self.schema.connect() as conn:
            run_cursor = conn.execute(
                """
                INSERT INTO memory_extraction_runs (
                    raw_message_id, model_provider, model_name, prompt_version,
                    input_hash, output_json, status
                )
                VALUES (?, ?, ?, ?, ?, ?, 'succeeded')
                """,
                (
                    raw_message_id,
                    self.chat_config.provider,
                    self.chat_config.model,
                    PROMPT_VERSION,
                    input_hash(raw_content),
                    output_json,
                ),
            )
            extraction_run_id = int(run_cursor.lastrowid)

            if should_remember:
                for item in memories:
                    duplicate = find_duplicate_memory(conn, item)
                    if duplicate is not None:
                        skipped_duplicates.append((clean_optional_text(item.get("title")) or short_text(str(item.get("content") or ""), 40), duplicate))
                        continue
                    memory_id = self._insert_memory(conn, raw_message_id, extraction_run_id, item)
                    memory_ids.append(memory_id)
                    topic_ids.extend(self._link_topics(conn, memory_id, item.get("topics") or []))
                    entity_ids.extend(self._link_entities(conn, memory_id, item.get("entities") or []))

            status = "processed" if should_remember else "ignored"
            conn.execute(
                """
                UPDATE raw_messages
                SET processed_status = ?, processed_at = datetime('now', 'localtime')
                WHERE id = ?
                """,
                (status, raw_message_id),
            )

        warning = None if should_remember else "model decided not to remember this input"
        if skipped_duplicates:
            detail = "; ".join(f"{title} ~= memory {memory_id}" for title, memory_id in skipped_duplicates[:5])
            warning = combine_warning(warning, f"skipped duplicate memory candidate(s): {detail}")
        if should_remember and not memory_ids and skipped_duplicates:
            warning = combine_warning(warning, "all extracted memory candidates were duplicates")

        return IngestResult(
            raw_message_id=raw_message_id,
            extraction_run_id=extraction_run_id,
            memory_ids=memory_ids,
            topic_ids=sorted(set(topic_ids)),
            entity_ids=sorted(set(entity_ids)),
            should_remember=should_remember,
            router_rebuilt=False,
            warning=warning,
        )

    def _insert_memory(
        self,
        conn: sqlite3.Connection,
        raw_message_id: int,
        extraction_run_id: int,
        item: dict[str, Any],
    ) -> int:
        content = clean_memory_text(clean_required_text(item.get("content"), "memory content"))
        title = clean_optional_text(item.get("title"))
        memory_category = normalize_memory_category(clean_optional_text(item.get("memory_category")))
        memory_type = clean_optional_text(item.get("memory_type")) or "other"
        importance = clamp_score(item.get("importance"), default=0.5)
        confidence = clamp_score(item.get("confidence"), default=0.7)
        cursor = conn.execute(
            """
            INSERT INTO memories (
                raw_message_id, extraction_run_id, content, title,
                memory_category, memory_type, importance, confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_message_id,
                extraction_run_id,
                content,
                title,
                memory_category,
                memory_type,
                importance,
                confidence,
            ),
        )
        return int(cursor.lastrowid)

    def _link_topics(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        topics: list[Any],
    ) -> list[int]:
        topic_ids: list[int] = []
        for topic in topics:
            if isinstance(topic, str):
                topic_data = {"name": topic}
            elif isinstance(topic, dict):
                topic_data = topic
            else:
                continue
            name = clean_optional_text(topic_data.get("name"))
            if not name:
                continue
            description = clean_optional_text(topic_data.get("description"))
            confidence = clamp_score(topic_data.get("confidence"), default=0.7)
            reason = clean_optional_text(topic_data.get("reason"))
            topic_id = upsert_topic(conn, name, description)
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_topics (memory_id, topic_id, confidence, reason)
                VALUES (?, ?, ?, ?)
                """,
                (memory_id, topic_id, confidence, reason),
            )
            topic_ids.append(topic_id)
        return topic_ids

    def _link_entities(
        self,
        conn: sqlite3.Connection,
        memory_id: int,
        entities: list[Any],
    ) -> list[int]:
        entity_ids: list[int] = []
        for entity in entities:
            if isinstance(entity, str):
                entity_data = {"name": entity, "type": "other"}
            elif isinstance(entity, dict):
                entity_data = entity
            else:
                continue
            name = clean_optional_text(entity_data.get("name"))
            if not name:
                continue
            entity_type = clean_optional_text(entity_data.get("type")) or clean_optional_text(entity_data.get("entity_type")) or "other"
            description = clean_optional_text(entity_data.get("description"))
            confidence = clamp_score(entity_data.get("confidence"), default=0.7)
            entity_id = upsert_entity(conn, name, entity_type, description)
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_entities (memory_id, entity_id, confidence)
                VALUES (?, ?, ?)
                """,
                (memory_id, entity_id, confidence),
            )
            entity_ids.append(entity_id)
        return entity_ids

    def _mark_raw_status(self, raw_message_id: int, status: str) -> None:
        with self.schema.connect() as conn:
            conn.execute(
                """
                UPDATE raw_messages
                SET processed_status = ?, processed_at = datetime('now', 'localtime')
                WHERE id = ?
                """,
                (status, raw_message_id),
            )


def input_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(cleaned[start : end + 1], strict=False)
    if not isinstance(payload, dict):
        raise ValueError("model output must be a JSON object")
    return payload


def preserve_exact_technical_tokens(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Restore a narrow set of code tokens that some chat models drop in JSON values."""
    atomic_memories = payload.get("atomic_memories")
    if not isinstance(atomic_memories, list) or not atomic_memories:
        return payload

    token_pattern = re.compile(
        r"(?i)(?<![\w.])json\.(?:loads?|dumps?)(?!\w)"
        r"|(?<![\w.])[A-Za-z_][\w-]*\.json(?!\w)"
        r"|(?<!\w)\.json(?!\w)"
    )
    source_tokens = list(dict.fromkeys(match.group(0) for match in token_pattern.finditer(content)))
    if not source_tokens:
        return payload

    source_lower = content.lower()

    def payload_text() -> str:
        return json.dumps(atomic_memories, ensure_ascii=False).lower()

    def restore_in_value(value: Any, token: str) -> Any:
        if isinstance(value, dict):
            return {key: restore_in_value(item, token) for key, item in value.items()}
        if isinstance(value, list):
            return [restore_in_value(item, token) for item in value]
        if not isinstance(value, str):
            return value

        lowered = token.lower()
        if lowered.startswith("json."):
            function_name = re.escape(token.split(".", 1)[1])
            variants = [rf"(?i)(?<![\w.])\.{function_name}(?!\w)"]
            if lowered == "json.load" and "json.loads" not in source_lower:
                variants.append(r"(?i)(?<![\w.])\.loads(?!\w)")
            for pattern in variants:
                value = re.sub(pattern, token, value)
            return value

        if lowered == ".json":
            return re.sub(r"(?<!\w)\.(?!\w)", token, value)

        base = re.escape(token[: -len(".json")])
        return re.sub(rf"(?i)\b{base}\.(?!\w)", token, value)

    for token in source_tokens:
        if token.lower() in payload_text():
            continue
        atomic_memories = restore_in_value(atomic_memories, token)
        payload["atomic_memories"] = atomic_memories

    missing = [token for token in source_tokens if token.lower() not in payload_text()]
    if missing and isinstance(atomic_memories[0], dict):
        current = str(atomic_memories[0].get("content") or "").rstrip()
        suffix = "原文中的精确技术标识：" + "、".join(missing) + "。"
        atomic_memories[0]["content"] = f"{current} {suffix}".strip()

    return payload


def preserve_personal_brain_feedback(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    if bool(payload.get("should_remember", False)):
        return payload
    if not looks_like_personal_brain_feedback(content):
        return payload

    forced = dict(payload)
    forced["should_remember"] = True
    forced["reason"] = (
        str(payload.get("reason") or "").strip()
        or "question-shaped Personal Brain product feedback should be preserved"
    )
    forced["atomic_memories"] = [
        {
            "title": classify_personal_brain_feedback_title(content),
            "content": rewrite_personal_brain_feedback_memory(content),
            "memory_category": classify_personal_brain_feedback_category(content),
            "memory_type": "idea",
            "importance": 0.72,
            "confidence": 0.68,
            "topics": [
                {
                    "name": "个人记忆系统改进",
                    "description": "围绕 Personal Brain 当前缺陷、近期修复和未来方向的产品反馈",
                    "confidence": 0.8,
                    "reason": "用户在讨论个人记忆系统的产品改进",
                }
            ],
            "entities": [
                {
                    "name": "Personal Brain",
                    "type": "product",
                    "description": "用户正在测试和改进的个人记忆系统",
                    "confidence": 0.9,
                }
            ],
        }
    ]
    return forced


def preserve_temporary_todo(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    if bool(payload.get("should_remember", False)):
        return payload
    if not looks_like_temporary_todo(content):
        return payload

    clean = " ".join(content.strip().split())
    forced = dict(payload)
    forced["should_remember"] = True
    forced["reason"] = (
        str(payload.get("reason") or "").strip()
        or "explicit short-term todo should be preserved as temporary todo"
    )
    forced["atomic_memories"] = [
        {
            "title": classify_temporary_todo_title(clean),
            "content": f"临时待办：{clean}",
            "memory_category": "临时待办",
            "memory_type": "plan",
            "importance": 0.55,
            "confidence": 0.72,
            "topics": [
                {
                    "name": "临时待办",
                    "description": "有时间边界或短期执行要求的提醒事项",
                    "confidence": 0.8,
                    "reason": "用户表达了明确的短期待办或提醒",
                }
            ],
            "entities": [],
        }
    ]
    return forced


def preserve_learning_note(content: str, payload: dict[str, Any]) -> dict[str, Any]:
    if bool(payload.get("should_remember", False)):
        return payload
    if not looks_like_learning_note(content):
        return payload

    clean = " ".join(content.strip().split())
    forced = dict(payload)
    forced["should_remember"] = True
    forced["reason"] = (
        str(payload.get("reason") or "").strip()
        or "compact reusable concept note should be preserved as learning"
    )
    forced["atomic_memories"] = [
        {
            "title": classify_learning_note_title(clean),
            "content": rewrite_learning_note_memory(clean),
            "memory_category": "学习",
            "memory_type": "fact",
            "importance": 0.62,
            "confidence": 0.72,
            "topics": [
                {
                    "name": "概念学习",
                    "description": "用户记录的短概念、定义、区别、类比或术语理解",
                    "confidence": 0.78,
                    "reason": "用户输入像可复习的概念学习笔记",
                }
            ],
            "entities": extract_learning_note_entities(clean),
        }
    ]
    return forced


def looks_like_learning_note(text: str) -> bool:
    clean = " ".join(text.strip().split())
    if not clean:
        return False
    if len(clean) < 8 or len(clean) > 180:
        return False
    if looks_like_temporary_todo(clean) or looks_like_personal_brain_feedback(clean):
        return False
    project_terms = ("Personal Brain", "个人记忆系统", "第二大脑", "质量报告", "消息入口")
    if any(term in clean for term in project_terms):
        return False
    if re.fullmatch(r"[\W_0-9a-zA-Z]+", clean) and not any(ch in clean for ch in ("+", "=", "：", ":")):
        return False

    concept_patterns = (
        "就是",
        "指的是",
        "定义",
        "区别",
        "类似于",
        "相当于",
        "可以理解为",
        "核心是",
        "本质是",
        "意思是",
        "等于",
    )
    if any(pattern in clean for pattern in concept_patterns):
        return True

    tech_concept_terms = (
        "memory",
        "recall",
        "embedding",
        "RAG",
        "agent",
        "prompt",
        "workflow",
        "harness",
        "模型",
        "微调",
        "长期记忆",
        "短期记忆",
    )
    return "+" in clean and any(term.lower() in clean.lower() for term in tech_concept_terms)


def classify_learning_note_title(text: str) -> str:
    subject = extract_learning_note_subject(text)
    if subject:
        return f"{subject}的学习理解"
    return "学习概念记录"


def rewrite_learning_note_memory(text: str) -> str:
    subject, explanation = split_learning_note_definition(text)
    if subject and explanation:
        return f"用户将“{subject}”理解为“{explanation}”，用于后续学习和复习。"
    return f"用户记录了一条学习概念：{text}"


def extract_learning_note_entities(text: str) -> list[dict[str, Any]]:
    subject = extract_learning_note_subject(text)
    if not subject:
        return []
    if len(subject) > 40:
        return []
    return [
        {
            "name": subject,
            "type": "concept",
            "description": "用户记录的学习概念",
            "confidence": 0.68,
        }
    ]


def extract_learning_note_subject(text: str) -> str | None:
    subject, _ = split_learning_note_definition(text)
    if subject:
        return subject
    if "+" in text:
        return text.split("，", 1)[0].split("。", 1)[0].strip()
    return None


def split_learning_note_definition(text: str) -> tuple[str | None, str | None]:
    separators = ("就是", "指的是", "可以理解为", "相当于", "类似于", "意思是", "等于")
    for sep in separators:
        if sep not in text:
            continue
        left, right = text.split(sep, 1)
        left = left.strip(" ：:，,。 ")
        right = right.strip(" ：:，,。 ")
        if left and right and len(left) <= 40:
            return left, right
    return None, None


def looks_like_temporary_todo(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    reminder_terms = ("别忘", "记得", "提醒", "待办", "要做", "准备", "打印", "联系")
    time_terms = (
        "今天",
        "明天",
        "后天",
        "下周",
        "周一",
        "周二",
        "周三",
        "周四",
        "周五",
        "周六",
        "周日",
        "会议前",
        "出发前",
        "之前",
        "截止",
    )
    if any(term in clean for term in reminder_terms) and any(term in clean for term in time_terms):
        return True
    return bool(re.search(r"\d{1,2}[月/-]\d{1,2}|周[一二三四五六日天]|星期[一二三四五六日天]", clean)) and any(
        term in clean for term in reminder_terms
    )


def classify_temporary_todo_title(text: str) -> str:
    if "会议" in text:
        return "会议临时待办"
    if "打印" in text:
        return "打印临时待办"
    if "联系" in text:
        return "联系临时待办"
    return "临时待办"


def looks_like_personal_brain_feedback(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    subject_terms = (
        "Personal Brain",
        "个人记忆系统",
        "第二大脑",
        "质量报告",
        "提取",
        "消息入口",
        "飞书",
        "embedding",
        "RAG",
        "检索",
    )
    signal_terms = (
        "怎么修",
        "怎么解决",
        "要不要",
        "需不需要",
        "是不是可以",
        "能不能",
        "可不可以",
        "应该",
        "优化",
        "改进",
        "问题",
        "缺陷",
        "失败",
        "重复",
        "不在线",
        "失效",
        "未来",
        "方向",
        "下一步",
    )
    extra_signal_terms = ("分类", "单独")
    return any(term.lower() in clean.lower() for term in subject_terms) and (
        any(term in clean for term in signal_terms)
        or any(term in clean for term in extra_signal_terms)
    )


def classify_personal_brain_feedback_category(text: str) -> str:
    if any(term in text for term in ("未来", "方向", "第二个我", "数字分身", "机器人", "接入")):
        return "未来产品设想"
    return "现有项目改进"


def classify_personal_brain_feedback_title(text: str) -> str:
    if any(term in text for term in ("失败", "不能", "不在线", "失效", "挂", "重复")):
        return "个人记忆系统当前缺陷反馈"
    if any(term in text for term in ("未来", "方向", "第二个我", "机器人")):
        return "个人记忆系统未来方向反馈"
    return "个人记忆系统近期改进反馈"


def rewrite_personal_brain_feedback_memory(text: str) -> str:
    clean = " ".join(text.strip().split())
    return f"用户提出一条 Personal Brain 产品反馈：{clean}"


def clean_required_text(value: Any, label: str) -> str:
    text = clean_optional_text(value)
    if not text:
        raise ValueError(f"{label} is required")
    return text


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def clean_memory_text(text: str) -> str:
    clean = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    clean = clean.replace("**", "").replace("`", "")
    clean = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", clean)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def find_duplicate_memory(conn: sqlite3.Connection, item: dict[str, Any]) -> int | None:
    candidate_content = clean_memory_text(clean_required_text(item.get("content"), "memory content"))
    candidate_title = clean_optional_text(item.get("title")) or ""
    candidate_category = normalize_memory_category(clean_optional_text(item.get("memory_category")))
    candidate_text = f"{candidate_title}\n{candidate_content}"
    candidate_key = normalize_for_near_duplicate(candidate_text)
    candidate_content_key = normalize_for_near_duplicate(candidate_content)
    if not candidate_key:
        return None
    rows = conn.execute(
        """
        SELECT id, title, content, memory_category
        FROM memories
        WHERE status = 'active'
        ORDER BY updated_at DESC, id DESC
        LIMIT 300
        """
    ).fetchall()
    for row in rows:
        existing_text = f"{row['title'] or ''}\n{row['content'] or ''}"
        existing_key = normalize_for_near_duplicate(existing_text)
        existing_content_key = normalize_for_near_duplicate(row["content"] or "")
        if not existing_key:
            continue
        if candidate_content_key and candidate_content_key == existing_content_key:
            return int(row["id"])
        if candidate_key == existing_key:
            return int(row["id"])
        if candidate_key in existing_key or existing_key in candidate_key:
            shorter = min(len(candidate_key), len(existing_key))
            longer = max(len(candidate_key), len(existing_key))
            if shorter >= 24 and shorter / max(1, longer) >= 0.72:
                return int(row["id"])
        similarity = SequenceMatcher(None, candidate_key, existing_key).ratio()
        if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
            return int(row["id"])
        if is_same_personal_brain_feedback_intent(
            candidate_text,
            candidate_category,
            existing_text,
            str(row["memory_category"] or ""),
        ):
            return int(row["id"])
    return None


def normalize_for_near_duplicate(text: str) -> str:
    clean = str(text).lower()
    clean = "".join(char for char in clean if char.isalnum())
    clean = clean.replace("个人记忆系统", "personalbrain")
    clean = clean.replace("这个记忆系统", "personalbrain")
    clean = clean.replace("第二大脑", "personalbrain")
    return clean[:500]


def is_same_personal_brain_feedback_intent(
    candidate_text: str,
    candidate_category: str,
    existing_text: str,
    existing_category: str,
) -> bool:
    if candidate_category != existing_category:
        return False
    if candidate_category not in {"现有项目改进", "未来产品设想"}:
        return False
    combined = f"{candidate_text}\n{existing_text}"
    if not any(term.lower() in combined.lower() for term in ("Personal Brain", "个人记忆系统", "第二大脑")):
        return False
    candidate_terms = personal_brain_feedback_terms(candidate_text)
    existing_terms = personal_brain_feedback_terms(existing_text)
    if not candidate_terms or not existing_terms:
        return False
    overlap = candidate_terms & existing_terms
    smaller = min(len(candidate_terms), len(existing_terms))
    return len(overlap) >= 4 and len(overlap) / max(1, smaller) >= 0.66


def personal_brain_feedback_terms(text: str) -> set[str]:
    term_groups = {
        "详情": ("详情", "详细", "完整", "完整内容", "详细视图"),
        "原始输入": ("原始输入", "原文", "raw", "依据"),
        "记忆内容": ("记忆内容", "内容", "摘要", "概要", "大概"),
        "查看": ("查看", "打开", "调出", "显示"),
        "编号": ("编号", "id", "记忆91", "memory"),
        "追溯": ("追溯", "来源", "核对", "准确性"),
        "去重": ("去重", "重复", "近重复", "降重"),
        "衰减": ("衰减", "降权", "过期", "时间", "时效"),
        "检索": ("检索", "召回", "问号", "唤醒"),
        "入口": ("入口", "飞书", "命令", "快捷"),
        "过度解读": ("过度", "建议", "发挥", "忠实"),
    }
    clean = text.lower()
    found: set[str] = set()
    for label, variants in term_groups.items():
        if any(variant.lower() in clean for variant in variants):
            found.add(label)
    return found


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def combine_warning(first: str | None, second: str | None) -> str | None:
    if first and second:
        return f"{first}; {second}"
    return first or second


def clamp_score(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def normalize_memory_category(value: str | None) -> str:
    if not value:
        return "其他"
    clean = value.strip()
    if clean in MEMORY_CATEGORIES:
        return clean
    aliases = {
        "项目改进": "现有项目改进",
        "当前项目改进": "现有项目改进",
        "产品设想": "未来产品设想",
        "未来方向": "未来产品设想",
        "产品技巧": "产品使用技巧",
        "使用技巧": "产品使用技巧",
        "自我认知": "自身认知更新",
        "认知更新": "自身认知更新",
        "学习记录": "学习",
        "学习笔记": "学习",
        "知识学习": "学习",
        "概念学习": "学习",
        "知识概念": "学习",
        "技术判断": "技术思考",
        "技术策略": "技术思考",
        "工作流": "工作流方法",
        "安全": "信息安全",
    }
    return aliases.get(clean, "其他")


def upsert_topic(conn: sqlite3.Connection, name: str, description: str | None) -> int:
    conn.execute(
        """
        INSERT INTO topics (name, description)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET
            description = COALESCE(excluded.description, topics.description),
            updated_at = datetime('now', 'localtime')
        """,
        (name, description),
    )
    row = conn.execute("SELECT id FROM topics WHERE name = ?", (name,)).fetchone()
    return int(row["id"])


def upsert_entity(
    conn: sqlite3.Connection,
    name: str,
    entity_type: str,
    description: str | None,
) -> int:
    conn.execute(
        """
        INSERT INTO entities (name, entity_type, description)
        VALUES (?, ?, ?)
        ON CONFLICT(name, entity_type) DO UPDATE SET
            description = COALESCE(excluded.description, entities.description)
        """,
        (name, entity_type, description),
    )
    row = conn.execute(
        "SELECT id FROM entities WHERE name = ? AND entity_type = ?",
        (name, entity_type),
    ).fetchone()
    return int(row["id"])
