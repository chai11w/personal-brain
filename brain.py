from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

from personal_brain import PersonalBrain
from personal_brain.answer import format_answer_result
from personal_brain.ask_debug import write_ask_debug_report
from personal_brain.daily_report import parse_report_date
from personal_brain.memory_view import format_memory_detail, format_memory_summary
from personal_brain.reviewer import format_memory_review
from personal_brain.semantic import format_recall_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal Brain V0 Memory Router")
    parser.add_argument("--config", default="config.json", help="config file path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init-db", help="initialize the AI-native database foundation")
    subparsers.add_parser("build-router", help="build brain_index.json and router manifests")
    subparsers.add_parser("stats", help="show local store stats")

    memory_list_parser = subparsers.add_parser("memory-list", help="list AI-generated memories for review")
    memory_list_parser.add_argument("--limit", type=int, default=20, help="max memories to show")

    memory_show_parser = subparsers.add_parser("memory-show", help="show one memory with raw evidence")
    memory_show_parser.add_argument("memory_id", type=int, help="memory id")

    memory_archive_parser = subparsers.add_parser("memory-archive", help="archive one memory by id")
    memory_archive_parser.add_argument("memory_id", type=int, help="memory id")

    interaction_list_parser = subparsers.add_parser("interaction-list", help="list recent Feishu/adapter interactions")
    interaction_list_parser.add_argument("--limit", type=int, default=20, help="max interactions to show")

    daily_report_parser = subparsers.add_parser("daily-report", help="write a local Markdown audit report for Codex review")
    daily_report_parser.add_argument("--date", default="today", help="today, yesterday, or YYYY-MM-DD")
    daily_report_parser.add_argument("--last-hours", type=int, help="write a rolling report for the previous N hours")
    daily_report_parser.add_argument("--output-dir", default="reports", help="directory for local Markdown reports")
    daily_report_parser.add_argument("--print-path-only", action="store_true", help="only print the output path")

    review_parser = subparsers.add_parser("review-memories", help="dry-run AI review of memory quality")
    review_parser.add_argument("--limit", type=int, default=80, help="max memories to review")
    review_parser.add_argument("--raw-message-id", type=int, help="review only memories extracted from one raw message")
    review_parser.add_argument("--output", help="optional JSON file path for the dry-run review")

    secure_add_parser = subparsers.add_parser("secure-add", help="add encrypted secure item")
    secure_add_parser.add_argument("--label", required=True, help="item label, for example GitHub main")
    secure_add_parser.add_argument("--type", default="password", help="secret type: password/api_key/token/note")
    secure_add_parser.add_argument("--username", help="optional username")
    secure_add_parser.add_argument("--note", help="optional non-secret note")

    subparsers.add_parser("secure-list", help="list secure item metadata without secrets")

    secure_get_parser = subparsers.add_parser("secure-get", help="decrypt one secure item")
    secure_get_parser.add_argument("label", help="item label")

    ingest_parser = subparsers.add_parser("ingest", help="extract AI atomic memories from input text")
    ingest_parser.add_argument("text", help="raw user input")
    ingest_parser.add_argument("--source", default="cli", help="message source")
    ingest_parser.add_argument("--sender", default="me", help="message sender")
    ingest_parser.add_argument(
        "--no-router",
        action="store_true",
        help="do not rebuild Memory Router after ingest",
    )

    embed_parser = subparsers.add_parser("embed-memories", help="embed active memories without embeddings")
    embed_parser.add_argument("--limit", type=int, default=100, help="max memories to embed")

    recall_parser = subparsers.add_parser("recall", help="semantic recall over embedded memories")
    recall_parser.add_argument("query", help="semantic query")
    recall_parser.add_argument("--limit", type=int, default=8, help="max recall results")

    ask_parser = subparsers.add_parser("ask", help="answer a question from recalled memory evidence")
    ask_parser.add_argument("question", help="question to answer")
    ask_parser.add_argument("--recall-limit", type=int, default=8, help="max memories to recall")
    ask_parser.add_argument("--evidence-limit", type=int, default=5, help="max evidence items to answer from")
    ask_parser.add_argument("--debug", action="store_true", help="write an ask debug Markdown report")
    ask_parser.add_argument(
        "--debug-output-dir",
        default="reports/ask-debug",
        help="directory for ask debug Markdown reports",
    )

    test_chat_parser = subparsers.add_parser("test-chat", help="test configured chat model")
    test_chat_parser.add_argument(
        "prompt",
        nargs="?",
        default="请用一句话回复：模型已接通。",
        help="test prompt",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    brain = PersonalBrain.from_config_file(args.config)

    if args.command == "init-db":
        result = brain.init_db()
        print(f"database: {result.database_path}")
        print(f"schema version: {result.schema_version}")
        print("tables:")
        for table, count in result.tables.items():
            print(f"- {table}: {count}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0

    if args.command == "build-router":
        result = brain.build_router()
        print(f"brain index: {result.brain_index_path}")
        print(f"topics: {result.topics_path} ({result.topic_count})")
        print(f"manifest: {result.manifest_path} ({result.memory_count})")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0

    if args.command == "ingest":
        result = brain.ingest(
            args.text,
            source=args.source,
            sender=args.sender,
            rebuild_router=not args.no_router,
        )
        print(f"raw_message_id: {result.raw_message_id}")
        print(f"extraction_run_id: {result.extraction_run_id}")
        print(f"should_remember: {result.should_remember}")
        print(f"memories: {result.memory_ids}")
        print(f"topics: {result.topic_ids}")
        print(f"entities: {result.entity_ids}")
        print(f"router_rebuilt: {result.router_rebuilt}")
        if result.input_route:
            print(f"input_type: {result.input_route['input_type']}")
            print(f"trigger_reason: {result.input_route['trigger_reason']}")
            print(f"original_input: {result.input_route['original_input']}")
        if result.warning:
            print(f"warning: {result.warning}")
        return 0

    if args.command == "stats":
        print(brain.stats())
        return 0

    if args.command == "memory-list":
        memories = brain.memory_list(limit=args.limit)
        if not memories:
            print("no AI memories yet")
            return 0
        for index, memory in enumerate(memories):
            if index:
                print("")
            print(format_memory_summary(memory))
        return 0

    if args.command == "memory-show":
        print(format_memory_detail(brain.memory_show(args.memory_id)))
        return 0

    if args.command == "memory-archive":
        result = brain.archive_memory(args.memory_id)
        title = result.title or short_text(result.content, 60)
        print(f"archived memory #{result.memory_id}: {title}")
        print(f"raw_message_id: {result.raw_message_id}")
        print(f"previous_status: {result.previous_status}")
        print(f"new_status: {result.new_status}")
        print(f"embeddings_deleted: {result.embeddings_deleted}")
        print("router_rebuilt: True")
        return 0

    if args.command == "interaction-list":
        interactions = brain.list_interactions(limit=args.limit)
        if not interactions:
            print("no interactions logged yet")
            return 0
        for index, item in enumerate(interactions):
            if index:
                print("")
            print(f"#{item['id']} {item['created_at']} source={item['source']} action={item['action']} status={item['status']} latency_ms={item['latency_ms']}")
            print(f"  user: {short_text(item['user_text'], 160)}")
            if item["reply_text"]:
                print(f"  reply: {short_text(item['reply_text'], 220)}")
            if item["error"]:
                print(f"  error: {item['error']}")
        return 0

    if args.command == "daily-report":
        if args.last_hours is not None:
            if args.last_hours <= 0:
                print("warning: --last-hours must be positive")
                return 1
            result = brain.recent_report(hours=args.last_hours, output_dir=Path(args.output_dir))
        else:
            try:
                report_date = parse_report_date(args.date)
            except ValueError:
                print("warning: --date must be today, yesterday, or YYYY-MM-DD")
                return 1
            result = brain.daily_report(report_date=report_date, output_dir=Path(args.output_dir))
        if args.print_path_only:
            print(result.output_path)
            return 0
        print(f"daily report: {result.output_path}")
        print(f"date: {result.report_date.isoformat()}")
        print(f"start_at: {result.start_at}")
        print(f"end_at: {result.end_at}")
        for key, value in result.counts.items():
            print(f"{key}: {value}")
        return 0

    if args.command == "review-memories":
        try:
            result = brain.review_memories(limit=args.limit, raw_message_id=args.raw_message_id)
        except Exception as exc:
            print(f"warning: {exc}")
            return 1
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(result.review_json, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"review json: {output_path}")
        print(format_memory_review(result))
        return 0

    if args.command == "embed-memories":
        result = brain.embed_missing_memories(limit=args.limit)
        print(f"embedded: {result.embedded_count}")
        print(f"skipped: {result.skipped_count}")
        if result.warning:
            print(f"warning: {result.warning}")
        return 0

    if args.command == "recall":
        try:
            results = brain.recall(args.query, limit=args.limit)
        except RuntimeError as exc:
            print(f"warning: {exc}")
            return 1
        if not results:
            print("no embedded memories found")
            return 0
        for index, result in enumerate(results):
            if index:
                print("")
            print(format_recall_result(result))
        return 0

    if args.command == "ask":
        try:
            result = brain.ask(
                args.question,
                recall_limit=args.recall_limit,
                evidence_limit=args.evidence_limit,
            )
        except RuntimeError as exc:
            print(f"warning: {exc}")
            return 1
        print(format_answer_result(result))
        if args.debug:
            debug_result = write_ask_debug_report(
                result=result,
                config=brain.config,
                recall_limit=args.recall_limit,
                evidence_limit=args.evidence_limit,
                output_dir=Path(args.debug_output_dir),
            )
            print("")
            print(f"ask debug report: {debug_result.output_path}")
        return 0

    if args.command == "secure-add":
        secret = getpass.getpass("secret: ")
        master_password = getpass.getpass("master password: ")
        item_id = brain.secure_add(
            label=args.label,
            secret_type=args.type,
            username=args.username,
            note=args.note,
            secret=secret,
            master_password=master_password,
        )
        print(f"secure item saved: {item_id}")
        print("secret was encrypted locally and was not sent to AI or Router")
        return 0

    if args.command == "secure-list":
        items = brain.secure_list()
        if not items:
            print("no secure items")
            return 0
        for item in items:
            username = f" username={item.username}" if item.username else ""
            note = f" note={item.note}" if item.note else ""
            print(f"#{item.id} {item.label} type={item.secret_type}{username}{note} updated={item.updated_at}")
        return 0

    if args.command == "secure-get":
        master_password = getpass.getpass("master password: ")
        item = brain.secure_get(args.label, master_password)
        print(f"label: {item.summary.label}")
        print(f"type: {item.summary.secret_type}")
        if item.summary.username:
            print(f"username: {item.summary.username}")
        print(f"secret: {item.secret}")
        return 0

    if args.command == "test-chat":
        print(brain.test_chat(args.prompt))
        return 0

    return 1


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def short_text(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "..."


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
