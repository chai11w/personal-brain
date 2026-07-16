from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from .extractor import parse_json_object
from .llm import LLMClient
from .semantic import RecallResult, SemanticMemory


@dataclass(frozen=True)
class RerankedEvidence:
    memory_id: int
    relevance: float
    reason: str
    recall: RecallResult


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    evidence: list[RerankedEvidence]
    warning: str | None = None
    recalled: list[RecallResult] | None = None


class AnswerEngine:
    """Evidence-constrained answer layer over semantic recall."""

    def __init__(self, semantic_memory: SemanticMemory, chat_model: LLMClient):
        self.semantic_memory = semantic_memory
        self.chat_model = chat_model

    def ask(self, question: str, recall_limit: int = 8, evidence_limit: int = 5) -> AnswerResult:
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("ask question cannot be empty")
        if not self.chat_model.available:
            raise RuntimeError("chat model unavailable; configure chat_model before ask")

        recalled = self.semantic_memory.recall(clean_question, limit=recall_limit)
        if not recalled:
            return AnswerResult(
                question=clean_question,
                answer="没有找到可用的相关记忆证据，所以我不能可靠回答这个问题。",
                evidence=[],
                warning="no recalled memories",
                recalled=[],
            )

        warning = None
        try:
            reranked = self._rerank(clean_question, recalled, evidence_limit=evidence_limit)
        except Exception as exc:
            warning = f"AI rerank failed; used semantic recall order: {exc}"
            reranked = [
                RerankedEvidence(
                    memory_id=item.memory_id,
                    relevance=item.score,
                    reason="Semantic recall fallback.",
                    recall=item,
                )
                for item in recalled[:evidence_limit]
            ]

        if not reranked:
            return AnswerResult(
                question=clean_question,
                answer="召回到了一些记忆，但没有足够相关的证据可以支撑回答。",
                evidence=[],
                warning=warning or "no relevant evidence after rerank",
                recalled=recalled,
            )

        answer = self._answer(clean_question, reranked)
        citation_warning = citation_contract_warning(answer, reranked)
        if citation_warning:
            warning = combine_warnings(warning, citation_warning)
        return AnswerResult(
            question=clean_question,
            answer=answer,
            evidence=reranked,
            warning=warning,
            recalled=recalled,
        )

    def _rerank(
        self,
        question: str,
        recalled: list[RecallResult],
        evidence_limit: int,
    ) -> list[RerankedEvidence]:
        evidence = [recall_to_payload(item) for item in recalled]
        prompt = {
            "task": "rerank_memory_evidence",
            "question": question,
            "current_date": date.today().isoformat(),
            "rules": [
                "Only judge the provided evidence.",
                "Prefer evidence that directly answers the question.",
                "For questions about today's tasks, prefer temporary todos, plans, reminders, and date-bound memories over product ideas about reminder features.",
                "For questions about today's tasks, do not select expired date-bound temporary todos just to fill max_results; it is acceptable to return fewer than max_results.",
                "Resolve relative words such as 今天, 明天, 下周 from the evidence created_at when possible.",
                "Set relevance between 0 and 1.",
                "Return at most the requested number of evidence items.",
                "Do not invent memory ids.",
            ],
            "max_results": evidence_limit,
            "evidence": evidence,
            "output_schema": {
                "selected": [
                    {
                        "memory_id": 1,
                        "relevance": 0.95,
                        "reason": "why this memory is relevant",
                    }
                ]
            },
        }
        answer = self.chat_model.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You rerank Personal Brain memory evidence. "
                        "Output JSON only. Do not add Markdown."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if not answer:
            raise RuntimeError("model returned empty rerank response")
        payload = parse_json_object(answer)
        selected = payload.get("selected") or []
        if not isinstance(selected, list):
            raise ValueError("rerank selected must be a list")

        by_id = {item.memory_id: item for item in recalled}
        reranked: list[RerankedEvidence] = []
        seen: set[int] = set()
        for item in selected:
            if not isinstance(item, dict):
                continue
            try:
                memory_id = int(item.get("memory_id"))
            except (TypeError, ValueError):
                continue
            if memory_id in seen or memory_id not in by_id:
                continue
            relevance = clamp_score(item.get("relevance"), default=0.0)
            if relevance <= 0.0:
                continue
            reranked.append(
                RerankedEvidence(
                    memory_id=memory_id,
                    relevance=relevance,
                    reason=str(item.get("reason") or "").strip(),
                    recall=by_id[memory_id],
                )
            )
            seen.add(memory_id)
            if len(reranked) >= evidence_limit:
                break
        reranked.sort(key=lambda item: item.relevance, reverse=True)
        return reranked

    def _answer(self, question: str, evidence: list[RerankedEvidence]) -> str:
        citation_entries = citation_entries_for_evidence(evidence)
        prompt = {
            "task": "answer_from_personal_memory_evidence",
            "question": question,
            "current_date": date.today().isoformat(),
            "rules": [
                "Answer only from the provided evidence.",
                "For questions about today's tasks, answer with concrete todos first, not product-feature discussions.",
                "For questions about today's tasks, do not present expired date-bound temporary todos as today's tasks. If only expired todos are available, say there is no clear current todo evidence and mention stale items only as items to confirm.",
                "Resolve relative words such as 今天, 明天, 下周 from the evidence created_at when possible.",
                "If a memory was created yesterday and says 明天, treat it as today.",
                "If an item is a product idea about supporting reminders, mention it only after concrete todos or omit it if concrete todos exist.",
                "Write for a human reader, not like a search report.",
                "Use natural concise Chinese unless the user asks otherwise.",
                "Write like a clear Feishu chat reply, not a formal report.",
                "Default structure: one short opening sentence, then a flat numbered list with bracketed citations, then one short judgment or suggestion if useful, then an evidence section.",
                "Do not use Markdown section headings such as ## or ###.",
                "Do not use bold pseudo-headings such as **工具与方法论补充**.",
                "Do not use bold, backticks, stars, decorative punctuation, or nested bullet lists.",
                "Avoid phrases like 具体方法可归纳为以下几类, 开发环境与流程优化, 核心策略是; sound simple and direct.",
                "Do not put memory_id/raw_message_id in headings.",
                "Avoid bullet-heavy or deeply nested Markdown. Prefer short paragraphs or a simple numbered list.",
                "If evidence explicitly separates multiple tools or methods, list them separately; do not merge distinct methods only to be concise.",
                "Each numbered item should usually be no more than two short lines.",
                "Use bracketed numeric citations in the answer body. Put the citation marker at the end of the specific claim it supports, for example: 项目决策应说明关键取舍[1].",
                "Each citation marker must come from citation_map. Do not invent markers.",
                "If one claim is supported by multiple evidence items, combine markers in one bracket, for example: [1,2].",
                "End with an evidence section using exact heading 依据： and one line per marker, for example: [1] memory 184 / raw 198.",
                "The evidence section must include every marker used in the answer body, and each line must include both memory id and raw id.",
                "Do not use the older single-line evidence style such as 依据：记忆1/原文2；记忆3/原文4.",
                "If evidence is thin, say what is uncertain.",
                "Do not use outside knowledge.",
            ],
            "citation_map": citation_entries,
            "evidence": [
                {
                    **recall_to_payload(item.recall),
                    "citation_marker": citation_entries[index]["marker"],
                    "rerank_relevance": item.relevance,
                    "rerank_reason": item.reason,
                }
                for index, item in enumerate(evidence)
            ],
        }
        answer = self.chat_model.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only provided Personal Brain evidence. "
                        "Optimize for readable Chinese chat replies. Keep each claim traceable "
                        "with bracketed numeric citation markers and a compact evidence table."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        if not answer:
            raise RuntimeError("model returned empty answer")
        return clean_answer_for_chat(answer.strip())


def recall_to_payload(item: RecallResult) -> dict[str, Any]:
    return {
        "memory_id": item.memory_id,
        "final_score": item.score,
        "semantic_score": item.semantic_score,
        "exact_match_boost": item.exact_match_boost,
        "same_day_todo_boost": item.same_day_todo_boost,
        "todo_lifecycle_adjustment": item.todo_lifecycle_adjustment,
        "title": item.title,
        "content": item.content,
        "memory_category": item.memory_category,
        "memory_type": item.memory_type,
        "importance": item.importance,
        "confidence": item.confidence,
        "created_at": item.created_at,
        "topics": item.topics,
        "raw_message_id": item.raw_message_id,
        "raw_content": item.raw_content,
    }


def clamp_score(value: Any, default: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def citation_entries_for_evidence(evidence: list[RerankedEvidence]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(evidence):
        marker = f"[{index + 1}]"
        entries.append(
            {
                "marker": marker,
                "memory_id": item.memory_id,
                "raw_message_id": item.recall.raw_message_id,
            }
        )
    return entries


def format_answer_result(result: AnswerResult) -> str:
    lines = [result.answer]
    if result.warning:
        lines.extend(["", f"warning: {result.warning}"])
    return "\n".join(lines)


def clean_answer_for_chat(answer: str) -> str:
    clean = answer.strip()
    clean = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", clean)
    clean = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", clean)
    clean = re.sub(r"`([^`\n]+)`", r"\1", clean)
    clean = clean.replace("**", "").replace("`", "")
    clean = clean.replace("证据：", "依据：").replace("证据:", "依据：")
    clean = re.sub(r"依据：[ \t]*(?=\[\d)", "依据：\n", clean)
    clean = re.sub(
        r"memory_id\s*=\s*(\d+)\s*/\s*raw_message_id\s*=\s*(\d+)",
        r"记忆\1/原文\2",
        clean,
    )
    clean = re.sub(
        r"memory_id\s*=\s*(\d+)\s*[,，;；]\s*raw_message_id\s*=\s*(\d+)",
        r"记忆\1/原文\2",
        clean,
    )
    clean = re.sub(r"(?m)^(\s*\d+[.、]\s*)\*+", r"\1", clean)
    clean = re.sub(r"\*+([：:])", r"\1", clean)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean)
    return clean.strip()


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def citation_contract_warning(answer: str, evidence: list[RerankedEvidence]) -> str | None:
    if not evidence:
        return None
    has_new_style_citation = any(
        f"memory {item.memory_id}" in answer and f"raw {item.recall.raw_message_id}" in answer
        for item in evidence
    )
    if has_new_style_citation:
        return None
    has_memory_citation = any(
        f"memory_id={item.memory_id}" in answer or f"记忆{item.memory_id}" in answer
        for item in evidence
    )
    has_raw_citation = any(
        f"raw_message_id={item.recall.raw_message_id}" in answer or f"原文{item.recall.raw_message_id}" in answer
        for item in evidence
    )
    if has_memory_citation and has_raw_citation:
        return None
    return "answer did not fully satisfy citation contract"


def combine_warnings(first: str | None, second: str | None) -> str | None:
    if first and second:
        return f"{first}; {second}"
    return first or second
