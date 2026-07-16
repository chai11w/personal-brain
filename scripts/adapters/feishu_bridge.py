from __future__ import annotations

import argparse
import json
import queue
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from personal_brain import PersonalBrain  # noqa: E402
from personal_brain.answer import AnswerResult  # noqa: E402
from personal_brain.llm import get_secret_env  # noqa: E402
from personal_brain.memory_view import MemoryDetail  # noqa: E402


FEISHU_OPEN_API = "https://open.feishu.cn/open-apis"


def safe_log(message: str, *, stream: Any = None) -> None:
    """Write diagnostics without allowing console encoding to affect delivery state."""
    target = sys.stdout if stream is None else stream
    try:
        print(message, file=target, flush=True)
    except UnicodeEncodeError:
        try:
            encoding = getattr(target, "encoding", None) or "ascii"
            escaped = message.encode(encoding, errors="backslashreplace").decode(
                encoding, errors="replace"
            )
            print(escaped, file=target, flush=True)
        except Exception:
            pass
    except Exception:
        pass


@dataclass(frozen=True)
class FeishuOptions:
    mode: str
    ask_prefix: str
    ack_message: str
    working_reaction: str | None
    verification_token: str | None
    app_id: str
    app_secret: str
    dry_run: bool
    max_message_age_seconds: int


@dataclass(frozen=True)
class BridgeReply:
    text: str
    action: str
    raw_message_id: int | None = None
    evidence: list[dict[str, Any]] | None = None


class FeishuClient:
    def __init__(self, app_id: str, app_secret: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self._tenant_access_token: str | None = None
        self._token_expires_at = 0.0
        self._lock = threading.Lock()

    def reply_text(self, message_id: str, text: str) -> None:
        token = self.tenant_access_token()
        body = {
            "msg_type": "text",
            "content": json.dumps({"text": text}, ensure_ascii=False),
        }
        request_json(
            f"{FEISHU_OPEN_API}/im/v1/messages/{message_id}/reply",
            method="POST",
            payload=body,
            headers={"Authorization": f"Bearer {token}"},
        )

    def add_reaction(self, message_id: str, emoji_type: str) -> None:
        token = self.tenant_access_token()
        body = {"reaction_type": {"emoji_type": emoji_type}}
        request_json(
            f"{FEISHU_OPEN_API}/im/v1/messages/{message_id}/reactions",
            method="POST",
            payload=body,
            headers={"Authorization": f"Bearer {token}"},
        )

    def tenant_access_token(self) -> str:
        now = time.time()
        with self._lock:
            if self._tenant_access_token and now < self._token_expires_at:
                return self._tenant_access_token
            data = request_json(
                f"{FEISHU_OPEN_API}/auth/v3/tenant_access_token/internal",
                method="POST",
                payload={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            token = str(data.get("tenant_access_token") or "")
            if not token:
                raise RuntimeError(f"failed to get tenant_access_token: {data}")
            expire = int(data.get("expire") or 7200)
            self._tenant_access_token = token
            self._token_expires_at = now + max(60, expire - 120)
            return token


class FeishuBrainBridge:
    def __init__(self, brain: PersonalBrain, client: FeishuClient, options: FeishuOptions):
        self.brain = brain
        self.client = client
        self.options = options
        self._seen_event_ids: set[str] = set()
        self._lock = threading.Lock()
        self._jobs: queue.Queue[tuple[int, str, str, str, bool] | None] = queue.Queue(maxsize=256)
        self._worker = threading.Thread(target=self._worker_loop, name="feishu-brain-worker", daemon=False)
        self._worker.start()
        self._recover_jobs()

    def handle_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "encrypt" in payload:
            return {"ok": False, "error": "encrypted events are not supported in MVP; leave Encrypt Key empty"}

        if is_url_verification(payload):
            if not self._valid_token(payload):
                return {"ok": False, "error": "invalid verification token"}
            return {"challenge": payload.get("challenge")}

        if not self._valid_token(payload):
            return {"ok": False, "error": "invalid verification token"}

        header = payload.get("header") or {}
        event_type = header.get("event_type") or payload.get("type")
        if event_type != "im.message.receive_v1":
            return {"ok": True, "ignored": event_type or "unknown"}

        event_id = str(header.get("event_id") or "")
        if event_id and event_id in self._seen_event_ids:
            return {"ok": True, "duplicate": event_id}
        if event_id:
            self._seen_event_ids.add(event_id)

        event = payload.get("event") or {}
        message = event.get("message") or {}
        message_id = str(message.get("message_id") or "")
        text = extract_text_message(message)
        sender = extract_sender(event)
        if not message_id or not text:
            return {"ok": True, "ignored": "non-text-or-missing-message"}

        message_created_at = extract_message_created_at(payload, event, message)
        stale = is_stale_message(message_created_at, self.options.max_message_age_seconds)
        try:
            interaction_id, created = self.brain.claim_interaction(
                message_id=message_id, source="feishu", sender=sender, user_text=text,
                mode=self.options.mode, delivery_required=not stale,
            )
        except Exception as exc:
            return {"ok": False, "unavailable": True, "error": f"interaction persistence failed: {exc}"}
        if not created:
            return {"ok": True, "duplicate_message": message_id}

        if stale:
            age_seconds = int(time.time() - message_created_at) if message_created_at else None
            note = f"stale Feishu message ignored; age_seconds={age_seconds}; no reply sent"
            self.brain.mark_interaction_ignored(interaction_id, action="stale_ignored", note=note)
            safe_log(f"ignored stale Feishu message_id={message_id} age_seconds={age_seconds}")
            return {"ok": True, "ignored": "stale-message", "message_id": message_id}

        self._mark_working_async(message_id)
        try:
            self._jobs.put_nowait((interaction_id, message_id, text, sender, False))
        except queue.Full:
            # The durable pending row remains recoverable after restart.
            return {"ok": True, "accepted": message_id, "queued": False}
        return {"ok": True, "accepted": message_id}

    def _worker_loop(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                self._process_and_reply(*job)
            finally:
                self._jobs.task_done()

    def _recover_jobs(self) -> None:
        for item in self.brain.recoverable_interactions():
            delivery_only = item["processing_status"] in {"succeeded", "failed"} and bool(item["reply_text"])
            try:
                self._jobs.put_nowait((
                    int(item["id"]), str(item["message_id"] or ""), str(item["user_text"]),
                    str(item["sender"]), delivery_only,
                ))
            except queue.Full:
                break

    def shutdown(self, timeout: float = 10.0) -> None:
        self._jobs.put(None, timeout=max(0.1, timeout))
        self._worker.join(timeout=max(0.1, timeout))

    def _process_and_reply(self, interaction_id: int, message_id: str, text: str, sender: str, delivery_only: bool = False) -> None:
        safe_log(f"received from {sender}: {text}")
        if delivery_only:
            item = self.brain.get_interaction(interaction_id)
            self._deliver(interaction_id, message_id, str(item["reply_text"]))
            return
        if not self.brain.claim_interaction_processing(interaction_id):
            return
        started_at = time.perf_counter()
        status = "succeeded"
        error = None
        bridge_reply: BridgeReply | None = None
        with self._lock:
            try:
                bridge_reply = self._reply_for_text(text, sender, message_id)
                reply = bridge_reply.text
            except Exception as exc:
                status = "failed"
                error = str(exc)
                reply = f"暂时处理失败：{exc}"
                bridge_reply = BridgeReply(text=reply, action="error")

        latency_ms = int((time.perf_counter() - started_at) * 1000)
        self.brain.save_interaction_reply(
            interaction_id, action=bridge_reply.action, reply_text=bridge_reply.text,
            raw_message_id=bridge_reply.raw_message_id, evidence=bridge_reply.evidence,
            processing_succeeded=status == "succeeded", error=error, latency_ms=latency_ms,
        )
        self._deliver(interaction_id, message_id, reply)

    def _deliver(self, interaction_id: int, message_id: str, reply: str) -> None:
        if self.options.dry_run:
            safe_log(f"dry-run reply to {message_id}: {reply}")
            self.brain.mark_interaction_delivery(interaction_id, succeeded=False, dry_run=True)
            return

        try:
            self.client.reply_text(message_id, reply)
            self.brain.mark_interaction_delivery(interaction_id, succeeded=True)
            safe_log(f"replied to {message_id}: {reply}")
        except Exception as exc:
            self.brain.mark_interaction_delivery(interaction_id, succeeded=False, error=str(exc))
            safe_log(f"failed to reply {message_id}: {exc}", stream=sys.stderr)

    def _reply_for_text(self, text: str, sender: str, message_id: str | None = None) -> BridgeReply:
        if is_help_command(text):
            return BridgeReply(text=format_help_reply(self.options.ask_prefix), action="help")

        detail_memory_id = extract_detail_memory_id(text)
        if detail_memory_id is not None:
            try:
                detail = self.brain.memory_show(detail_memory_id)
            except KeyError:
                return BridgeReply(text=f"没有找到记忆 #{detail_memory_id}。", action="detail")
            return BridgeReply(
                text=format_memory_detail_reply(detail),
                action="detail",
                raw_message_id=detail.raw_message_id,
            )

        archive_memory_id = extract_archive_memory_id(text)
        if archive_memory_id is not None:
            result = self.brain.archive_memory(archive_memory_id)
            title = result.title or short_text(result.content, 48)
            reply = (
                f"已作废记忆 #{result.memory_id}：{title}\n"
                f"这条记忆不会再进入语义召回或 Router，但原始证据仍保留。"
            )
            if result.previous_status == result.new_status:
                reply = f"记忆 #{result.memory_id} 之前已经是作废状态。\n{reply}"
            return BridgeReply(
                text=reply,
                action="archive",
                raw_message_id=result.raw_message_id,
            )
        if self.options.mode == "remember":
            return self._remember_text(text, sender, message_id)
        if self.options.mode == "ask":
            result = self.brain.ask(text)
            return BridgeReply(text=result.answer, action="ask", evidence=answer_evidence_payload(result))
        if self.options.mode == "auto":
            question = extract_question(text, self.options.ask_prefix)
            if question:
                result = self.brain.ask(question)
                return BridgeReply(text=result.answer, action="ask", evidence=answer_evidence_payload(result))
            return self._remember_text(text, sender, message_id)
        raise ValueError(f"unsupported mode: {self.options.mode}")

    def _mark_working_async(self, message_id: str) -> None:
        if not self.options.working_reaction or self.options.dry_run:
            return
        thread = threading.Thread(
            target=self._mark_working,
            args=(message_id,),
            daemon=True,
        )
        thread.start()

    def _mark_working(self, message_id: str) -> None:
        emoji_type = self.options.working_reaction
        if not emoji_type or self.options.dry_run:
            return
        try:
            self.client.add_reaction(message_id, emoji_type)
            safe_log(f"reacted to {message_id}: {emoji_type}")
        except Exception as exc:
            safe_log(f"failed to react {message_id}: {exc}", stream=sys.stderr)

    def _remember_text(self, text: str, sender: str, message_id: str | None = None) -> BridgeReply:
        result = self.brain.ingest(
            text, sender=sender, source="feishu", source_message_id=message_id
        )
        if result.memory_ids:
            details = [self.brain.memory_show(memory_id) for memory_id in result.memory_ids]
            reply = format_remembered_reply(self.options.ack_message, details)
            if result.warning:
                reply = f"{reply}\n\n提醒：{result.warning}"
            return BridgeReply(text=reply, action="remember", raw_message_id=result.raw_message_id)
        if result.should_remember:
            return BridgeReply(
                text="已处理，这次没有新增长期记忆；可能是候选已存在或没有产生新条目。",
                action="received",
                raw_message_id=result.raw_message_id,
            )
        if result.warning:
            return BridgeReply(
                text="已收到，不过这句没有写入长期记忆。",
                action="ignored",
                raw_message_id=result.raw_message_id,
            )
        return BridgeReply(
            text=self.options.ack_message or "Personal Brain 已记住。",
            action="received",
            raw_message_id=result.raw_message_id,
        )

    def _record_interaction(
        self,
        *,
        message_id: str,
        text: str,
        sender: str,
        reply: BridgeReply,
        status: str,
        error: str | None,
        latency_ms: int,
    ) -> None:
        try:
            self.brain.record_interaction(
                message_id=message_id,
                source="feishu",
                sender=sender,
                user_text=text,
                mode=self.options.mode,
                action=reply.action,
                raw_message_id=reply.raw_message_id,
                reply_text=reply.text,
                evidence=reply.evidence,
                status=status,
                error=error,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            safe_log(f"failed to record interaction log {message_id}: {exc}", stream=sys.stderr)

    def _record_stale_interaction(
        self,
        *,
        message_id: str,
        text: str,
        sender: str,
        age_seconds: int | None,
    ) -> None:
        if age_seconds is None:
            note = "stale Feishu message ignored before processing; no reply sent"
        else:
            note = f"stale Feishu message ignored before processing; age_seconds={age_seconds}; no reply sent"
        try:
            self.brain.record_interaction(
                message_id=message_id,
                source="feishu",
                sender=sender,
                user_text=text,
                mode=self.options.mode,
                action="stale_ignored",
                raw_message_id=None,
                reply_text=note,
                evidence=None,
                status="succeeded",
                error=None,
                latency_ms=0,
            )
        except Exception as exc:
            safe_log(f"failed to record stale interaction log {message_id}: {exc}", stream=sys.stderr)

    def _valid_token(self, payload: dict[str, Any]) -> bool:
        expected = self.options.verification_token
        if not expected:
            return True
        actual = (payload.get("header") or {}).get("token") or payload.get("token")
        return actual == expected


class FeishuHandler(BaseHTTPRequestHandler):
    bridge: FeishuBrainBridge

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json({"ok": True})
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json({"error": "invalid json"}, status=400)
            return

        if self.path not in {"/feishu/events", "/"}:
            self._send_json({"error": "not found"}, status=404)
            return

        result = self.bridge.handle_payload(payload)
        status = 503 if result.get("unavailable") else (200 if result.get("ok", True) else 403)
        self._send_json(result, status=status)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Brain Feishu bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=["remember", "ask", "auto"], default="auto")
    parser.add_argument("--ask-prefix", default="?")
    parser.add_argument("--ack-message", default="Personal Brain 已记住。")
    parser.add_argument(
        "--working-reaction",
        default="OK",
        help="emoji_type reaction added immediately when a message is accepted; empty disables it",
    )
    parser.add_argument("--app-id-env", default="FEISHU_APP_ID")
    parser.add_argument("--app-secret-env", default="FEISHU_APP_SECRET")
    parser.add_argument("--verification-token-env", default="FEISHU_VERIFICATION_TOKEN")
    parser.add_argument("--dry-run", action="store_true", help="process events but do not call Feishu reply API")
    parser.add_argument(
        "--max-message-age-minutes",
        type=int,
        default=15,
        help="ignore Feishu messages older than this many minutes; 0 disables stale-message filtering",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    app_id = require_secret_env(args.app_id_env)
    app_secret = require_secret_env(args.app_secret_env)
    verification_token = get_secret_env(args.verification_token_env)

    brain = PersonalBrain.from_config_file(args.config)
    brain.init_db()
    client = FeishuClient(app_id=app_id, app_secret=app_secret)
    options = FeishuOptions(
        mode=args.mode,
        ask_prefix=args.ask_prefix,
        ack_message=args.ack_message,
        working_reaction=args.working_reaction or None,
        verification_token=verification_token,
        app_id=app_id,
        app_secret=app_secret,
        dry_run=args.dry_run,
        max_message_age_seconds=max(0, args.max_message_age_minutes) * 60,
    )
    bridge = FeishuBrainBridge(brain=brain, client=client, options=options)
    FeishuHandler.bridge = bridge
    server = ThreadingHTTPServer((args.host, args.port), FeishuHandler)
    safe_log(f"Feishu bridge listening on http://{args.host}:{args.port}/feishu/events")
    safe_log(f"mode={args.mode} ask_prefix={args.ask_prefix!r} dry_run={args.dry_run}")
    try:
        server.serve_forever()
    finally:
        server.shutdown()
        server.server_close()
        bridge.shutdown(timeout=10.0)
    return 0


def is_url_verification(payload: dict[str, Any]) -> bool:
    payload_type = payload.get("type") or (payload.get("header") or {}).get("type")
    return payload_type == "url_verification" and "challenge" in payload


def extract_text_message(message: dict[str, Any]) -> str:
    if message.get("message_type") != "text":
        return ""
    try:
        content = json.loads(str(message.get("content") or "{}"))
    except json.JSONDecodeError:
        return ""
    return str(content.get("text") or "").strip()


def extract_sender(event: dict[str, Any]) -> str:
    sender = event.get("sender") or {}
    sender_id = sender.get("sender_id") or {}
    for key in ("user_id", "open_id", "union_id"):
        value = sender_id.get(key)
        if value:
            return str(value)
    return "feishu"


def extract_message_created_at(
    payload: dict[str, Any],
    event: dict[str, Any],
    message: dict[str, Any],
) -> float | None:
    header = payload.get("header") or {}
    candidates = [
        message.get("create_time"),
        message.get("update_time"),
        event.get("create_time"),
        header.get("create_time"),
        header.get("event_create_time"),
        header.get("timestamp"),
    ]
    for candidate in candidates:
        timestamp = parse_feishu_timestamp(candidate)
        if timestamp is not None:
            return timestamp
    return None


def parse_feishu_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    if timestamp > 10_000_000_000_000:
        return timestamp / 1_000_000
    if timestamp > 10_000_000_000:
        return timestamp / 1000
    return timestamp


def is_stale_message(created_at: float | None, max_age_seconds: int) -> bool:
    if created_at is None or max_age_seconds <= 0:
        return False
    return time.time() - created_at > max_age_seconds


def extract_question(text: str, ask_prefix: str) -> str | None:
    clean = text.strip()
    prefixes = [ask_prefix]
    if ask_prefix == "?":
        prefixes.append("？")
    for prefix in prefixes:
        if prefix and clean.startswith(prefix):
            question = clean[len(prefix) :].strip()
            return question or None
    return None


def is_help_command(text: str) -> bool:
    clean = text.strip().lower()
    return clean in {"!", "！"}


def extract_detail_memory_id(text: str) -> int | None:
    clean = text.strip()
    match = re.fullmatch(r"#\s*(\d+)", clean)
    if not match:
        return None
    memory_id = int(match.group(1))
    return memory_id if memory_id > 0 else None


def extract_archive_memory_id(text: str) -> int | None:
    clean = text.strip()
    match = re.fullmatch(r"-\s*(\d+)", clean)
    if not match:
        return None
    memory_id = int(match.group(1))
    return memory_id if memory_id > 0 else None


def format_help_reply(ask_prefix: str) -> str:
    question_prefix = "？" if ask_prefix == "?" else ask_prefix
    return "\n".join(
        [
            "Personal Brain 快捷指令：",
            "",
            "普通发送：记住一条内容，AI 会判断是否值得长期记忆。",
            f"{question_prefix}问题：从已有记忆里检索并回答，例如：{question_prefix}我之前对 Personal Brain 有什么看法？",
            "#91：查看某条记忆的原始输入和实际存入内容。",
            "-91：作废某条记忆；原始输入和审计记录仍保留。",
            "!：显示这份快捷指令。",
        ]
    )


def format_memory_detail_reply(detail: MemoryDetail) -> str:
    topics = "、".join(detail.summary.topics) if detail.summary.topics else "无"
    entities = "、".join(detail.entities) if detail.entities else "无"
    topic_reasons = detail.topic_reasons[:3]
    lines = [
        f"记忆 #{detail.summary.id}",
        "",
        "你输入的是：",
        f"raw_message_id：{detail.raw_message_id}",
        f"来源：{detail.raw_source} / {detail.raw_sender} / {detail.raw_created_at}",
        detail.raw_content,
        "",
        "Personal Brain 保存为：",
        f"标题：{detail.summary.title or '无标题'}",
        f"大类：{detail.summary.memory_category}",
        f"类型：{detail.summary.memory_type}",
        f"重要度/置信度：{detail.summary.importance:.2f}/{detail.summary.confidence:.2f}",
        f"主题：{topics}",
        f"实体：{entities}",
        detail.summary.content,
    ]
    if topic_reasons:
        lines.extend(["", "主题依据："])
        lines.extend(f"- {reason}" for reason in topic_reasons)
    lines.extend(
        [
            "",
            "提取记录：",
            f"extraction_run_id：{detail.extraction_run_id}",
            f"模型：{detail.model_provider}/{detail.model_name}",
            f"prompt_version：{detail.prompt_version}",
            f"状态：{detail.extraction_status}",
        ]
    )
    return "\n".join(
        lines
    )


def format_remembered_reply(ack_message: str, details: list[MemoryDetail]) -> str:
    lines = [ack_message or "Personal Brain 已记住。"]
    if len(details) >= 4:
        lines.append(f"这句话被拆成了 {len(details)} 条记忆，后续可以复盘是否拆得过细。")
    for index, detail in enumerate(details, start=1):
        topics = "、".join(detail.summary.topics) if detail.summary.topics else "无主题"
        lines.extend(
            [
                "",
                f"{index}. 记忆ID：{detail.summary.id}",
                f"   大类：{detail.summary.memory_category}",
                f"   主题：{topics}",
                f"   内容：{detail.summary.content}",
            ]
        )
    return "\n".join(lines)


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


def answer_evidence_payload(result: AnswerResult) -> list[dict[str, Any]]:
    return [
        {
            "memory_id": item.memory_id,
            "raw_message_id": item.recall.raw_message_id,
            "relevance": item.relevance,
            "title": item.recall.title,
            "memory_category": item.recall.memory_category,
        }
        for item in result.evidence
    ]


def request_json(
    url: str,
    method: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Feishu HTTP error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError) as exc:
            last_error = exc
            if attempt == 3:
                raise RuntimeError(f"Feishu request failed after retries: {exc}") from exc
            time.sleep(0.8 * attempt)
    else:
        raise RuntimeError(f"Feishu request failed: {last_error}")
    code = data.get("code", 0)
    if code not in {0, "0"}:
        raise RuntimeError(f"Feishu API error: {data}")
    return data


def require_secret_env(name: str) -> str:
    value = get_secret_env(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
