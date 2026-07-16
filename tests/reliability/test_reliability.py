from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from personal_brain.brain import PersonalBrain
from personal_brain.config import BrainConfig, ChatModelConfig, EmbeddingModelConfig
from personal_brain.extractor import MemoryExtractor
from personal_brain.router import MemoryRouterBuilder, load_router_bundle
from personal_brain.schema import BrainSchema, ClosingConnection, SCHEMA_VERSION
from scripts.adapters.feishu_bridge import BridgeReply, FeishuBrainBridge, FeishuOptions


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_PATHS = [
    ROOT / "data" / "personal_brain.sqlite3",
    ROOT / "brain_index.json",
    ROOT / "memory" / "topics.json",
    ROOT / "memory" / "memory_manifest.json",
    ROOT / "config.json",
]


def fingerprint(path: Path):
    if not path.exists():
        return None
    stat = path.stat()
    return hashlib.sha256(path.read_bytes()).hexdigest(), stat.st_mtime_ns, stat.st_size


def config_for(root: Path) -> BrainConfig:
    return BrainConfig(
        database_path=root / "data" / "test.sqlite3",
        memory_dir=root / "router",
        brain_index_path=root / "brain_index.json",
        chat_model=ChatModelConfig(enabled=False),
        embedding_model=EmbeddingModelConfig(enabled=False),
    )


class FakeChat:
    def __init__(self, responses):
        self.responses = list(responses)

    @property
    def available(self):
        return True

    def chat(self, *args, **kwargs):
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeClient:
    def __init__(self, fail=False):
        self.fail = fail
        self.replies = []

    def reply_text(self, message_id, text):
        if self.fail:
            raise RuntimeError("synthetic delivery failure")
        self.replies.append((message_id, text))

    def add_reaction(self, message_id, emoji):
        return None


def options(dry_run=False):
    return FeishuOptions(
        mode="remember", ask_prefix="?", ack_message="ok", working_reaction=None,
        verification_token=None, app_id="synthetic", app_secret="synthetic",
        dry_run=dry_run, max_message_age_seconds=0,
    )


def payload(message_id="m-1", event_id="e-1", text="synthetic text"):
    return {
        "header": {"event_type": "im.message.receive_v1", "event_id": event_id},
        "event": {
            "message": {"message_id": message_id, "message_type": "text", "content": json.dumps({"text": text})},
            "sender": {"sender_id": {"open_id": "synthetic-user"}},
        },
    }


class ReliabilityTests(unittest.TestCase):
    def make_bridge(self, brain, client):
        bridge = FeishuBrainBridge(brain, client, options())
        self.addCleanup(lambda: bridge.shutdown(timeout=1.0) if bridge._worker.is_alive() else None)
        return bridge

    @classmethod
    def setUpClass(cls):
        cls.production_before = {path: fingerprint(path) for path in PRODUCTION_PATHS}

    @classmethod
    def tearDownClass(cls):
        after = {path: fingerprint(path) for path in PRODUCTION_PATHS}
        if after != cls.production_before:
            raise AssertionError("reliability tests changed a protected production file")

    def test_a1_a10_schema_migration_backup_wal_full_and_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "v2.sqlite3"
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                conn.executescript(
                    """
                    CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
                    INSERT INTO schema_migrations VALUES(2, 'synthetic');
                    CREATE TABLE raw_messages(id INTEGER PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
                      sender TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, metadata_json TEXT,
                      processed_status TEXT DEFAULT 'pending', processed_at TEXT);
                    INSERT INTO raw_messages(id,content,source,sender) VALUES(1,'fixture','test','user');
                    """
                )
            result = BrainSchema(db).initialize()
            self.assertEqual(result.schema_version, SCHEMA_VERSION)
            backups = list(root.glob("*.bak"))
            self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0], factory=ClosingConnection) as backup:
                self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(backup.execute("SELECT content FROM raw_messages WHERE id=1").fetchone()[0], "fixture")
            schema = BrainSchema(db)
            with schema.connect_write() as conn:
                self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
                self.assertEqual(conn.execute("PRAGMA synchronous").fetchone()[0], 2)
                self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0], 1)

    def test_a7_readonly_missing_and_existing_do_not_write(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "missing" / "brain.sqlite3"
            schema = BrainSchema(db)
            with self.assertRaises(FileNotFoundError):
                schema.stats()
            self.assertFalse(db.parent.exists())
            schema.initialize()
            before = fingerprint(db)
            with schema.connect_readonly() as conn:
                self.assertEqual(conn.execute("PRAGMA query_only").fetchone()[0], 1)
            self.assertEqual(fingerprint(db), before)
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")

    def test_a10_migration_failure_rolls_back_additive_ddl_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "v2.sqlite3"
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                conn.executescript(
                    """
                    CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
                    INSERT INTO schema_migrations VALUES(2, 'synthetic');
                    CREATE TABLE raw_messages(id INTEGER PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
                      sender TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP, metadata_json TEXT,
                      processed_status TEXT DEFAULT 'pending', processed_at TEXT);
                    INSERT INTO raw_messages(id,content,source,sender) VALUES(1,'survives','test','user');
                    """
                )
            with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                BrainSchema(db).initialize(fail_at="after_additive_migration")
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                self.assertEqual(conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0], 2)
                columns = {row[1] for row in conn.execute("PRAGMA table_info(raw_messages)")}
                self.assertNotIn("source_message_id", columns)
                self.assertEqual(conn.execute("SELECT content FROM raw_messages WHERE id=1").fetchone()[0], "survives")
            backups = list(root.glob("*.bak"))
            self.assertEqual(len(backups), 1)
            with sqlite3.connect(backups[0], factory=ClosingConnection) as conn:
                self.assertEqual(conn.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_a3_atomic_interaction_and_raw_claims_under_concurrency(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            brain = PersonalBrain(config_for(root))
            brain.init_db()
            with ThreadPoolExecutor(max_workers=20) as pool:
                claims = list(pool.map(
                    lambda _: brain.claim_interaction(
                        message_id="same", source="feishu", sender="u", user_text="x", mode="remember"
                    ), range(20)
                ))
            self.assertEqual(sum(created for _, created in claims), 1)
            extractor = MemoryExtractor(brain.schema, FakeChat([]), ChatModelConfig())
            with ThreadPoolExecutor(max_workers=20) as pool:
                raw_claims = list(pool.map(
                    lambda _: extractor.capture_raw("x", "feishu", "u", source_message_id="same"), range(20)
                ))
            self.assertEqual(sum(created for _, created in raw_claims), 1)
            self.assertEqual(len({raw_id for raw_id, _ in raw_claims}), 1)

    def test_a6_failed_raw_reprocesses_same_row_and_single_concurrent_claim(self):
        success = json.dumps({"should_remember": False, "atomic_memories": []})
        with tempfile.TemporaryDirectory() as td:
            schema = BrainSchema(Path(td) / "brain.sqlite3")
            schema.initialize()
            extractor = MemoryExtractor(schema, FakeChat([RuntimeError("model down"), success]), ChatModelConfig())
            raw_id, _ = extractor.capture_raw("fixture", "test", "u", source_message_id="one")
            with self.assertRaises(RuntimeError):
                extractor.process_raw(raw_id)
            with schema.connect_readonly() as conn:
                self.assertEqual(conn.execute("SELECT processed_status FROM raw_messages WHERE id=?", (raw_id,)).fetchone()[0], "failed")
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM memory_extraction_runs WHERE raw_message_id=?", (raw_id,)).fetchone()[0], 1)
            result = extractor.reprocess(raw_id)
            self.assertEqual(result.raw_message_id, raw_id)
            with schema.connect_readonly() as conn:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM memory_extraction_runs").fetchone()[0], 2)
            with self.assertRaises(RuntimeError):
                extractor.reprocess(raw_id)

    def test_a2_a5_bridge_persists_before_ack_and_delivery_is_truthful(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            client = FakeClient(fail=True)
            bridge = self.make_bridge(brain, client)
            bridge._reply_for_text = lambda text, sender, message_id=None: BridgeReply("reply", "synthetic")
            result = bridge.handle_payload(payload())
            self.assertEqual(result["accepted"], "m-1")
            with brain.schema.connect_readonly() as conn:
                self.assertEqual(conn.execute("SELECT status FROM interaction_logs").fetchone()[0], "accepted")
            bridge._jobs.join()
            row = brain.list_interactions(1)[0]
            self.assertEqual(row["processing_status"], "succeeded")
            self.assertEqual(row["delivery_status"], "failed")
            self.assertEqual(row["status"], "delivery_failed")
            self.assertIsNone(row["delivered_at"])
            bridge.shutdown()
            brain.prepare_interaction_retry(int(row["id"]))
            retry_client = FakeClient()
            retry_bridge = self.make_bridge(brain, retry_client)
            retry_bridge._jobs.join()
            self.assertEqual(retry_client.replies, [("m-1", "reply")])
            retried = brain.get_interaction(int(row["id"]))
            self.assertEqual(retried["delivery_status"], "succeeded")
            self.assertEqual(retried["attempt_count"], 1)
            retry_bridge.shutdown()

    def test_a4_pending_claim_recovers_without_reprocessing_reply_ready(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            interaction_id, _ = brain.claim_interaction(
                message_id="recover", source="feishu", sender="u", user_text="x", mode="remember"
            )
            brain.claim_interaction_processing(interaction_id)
            brain.save_interaction_reply(interaction_id, action="synthetic", reply_text="saved")
            client = FakeClient()
            bridge = self.make_bridge(brain, client)
            bridge._jobs.join()
            self.assertEqual(client.replies, [("recover", "saved")])
            row = brain.get_interaction(interaction_id)
            self.assertEqual(row["delivery_status"], "succeeded")
            bridge.shutdown()

    def test_a4_pending_claim_is_processed_on_new_bridge(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            interaction_id, _ = brain.claim_interaction(
                message_id="pending-recover", source="feishu", sender="u", user_text="x", mode="remember"
            )
            client = FakeClient()
            original = FeishuBrainBridge._reply_for_text
            FeishuBrainBridge._reply_for_text = lambda self, text, sender, message_id=None: BridgeReply("processed", "synthetic")
            try:
                bridge = self.make_bridge(brain, client)
                bridge._jobs.join()
            finally:
                FeishuBrainBridge._reply_for_text = original
            row = brain.get_interaction(interaction_id)
            self.assertEqual(row["processing_status"], "succeeded")
            self.assertEqual(row["delivery_status"], "succeeded")
            self.assertEqual(client.replies, [("pending-recover", "processed")])
            bridge.shutdown()

    def test_a2_a3_webhook_concurrency_has_one_processing_and_reply(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            client = FakeClient()
            bridge = self.make_bridge(brain, client)
            calls = 0
            lock = threading.Lock()

            def reply_once(text, sender, message_id=None):
                nonlocal calls
                with lock:
                    calls += 1
                return BridgeReply("one", "synthetic")

            bridge._reply_for_text = reply_once
            with ThreadPoolExecutor(max_workers=20) as pool:
                results = list(pool.map(
                    lambda i: bridge.handle_payload(payload("same-webhook", f"event-{i}")), range(20)
                ))
            bridge._jobs.join()
            self.assertEqual(sum("accepted" in result for result in results), 1)
            self.assertEqual(calls, 1)
            self.assertEqual(client.replies, [("same-webhook", "one")])
            self.assertEqual(len(brain.list_interactions(50)), 1)
            self.assertFalse(bridge._worker.daemon)
            bridge.shutdown()
            self.assertFalse(bridge._worker.is_alive())

    def test_a2_claim_failure_is_not_acknowledged(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            bridge = self.make_bridge(brain, FakeClient())
            brain.claim_interaction = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("db unavailable"))
            result = bridge.handle_payload(payload("fail", "failure-event"))
            self.assertFalse(result["ok"])
            self.assertTrue(result["unavailable"])
            self.assertNotIn("accepted", result)
            bridge.shutdown()

    def test_a8_router_publish_is_atomic_and_checksums_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            schema = BrainSchema(root / "brain.sqlite3")
            schema.initialize()
            builder = MemoryRouterBuilder(schema.database_path, root / "memory", root / "brain_index.json")
            builder.build()
            old = (root / "brain_index.json").read_bytes()
            for point in ("after_first_file", "before_generation_rename", "before_pointer_replace"):
                with self.assertRaises(RuntimeError):
                    builder.build(fail_at=point)
                self.assertEqual((root / "brain_index.json").read_bytes(), old)
            result = builder.build()
            pointer = json.loads(result.brain_index_path.read_text(encoding="utf-8"))
            topics = Path(pointer["entrypoints"]["topics"])
            manifest = Path(pointer["entrypoints"]["memory_manifest"])
            self.assertEqual(json.loads(topics.read_text(encoding="utf-8"))["generation_id"], pointer["generation_id"])
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["generation_id"], pointer["generation_id"])
            self.assertEqual(hashlib.sha256(topics.read_bytes()).hexdigest(), pointer["checksums"]["topics.json"])
            self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), pointer["checksums"]["memory_manifest.json"])
            loaded_pointer, loaded_topics, loaded_manifest = load_router_bundle(result.brain_index_path)
            self.assertEqual(loaded_pointer["generation_id"], loaded_topics["generation_id"])
            self.assertEqual(loaded_pointer["generation_id"], loaded_manifest["generation_id"])


if __name__ == "__main__":
    unittest.main()
