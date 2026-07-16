from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import brain as brain_cli
from personal_brain.brain import PersonalBrain
from personal_brain.config import BrainConfig, ChatModelConfig, EmbeddingModelConfig
from personal_brain.extractor import MemoryExtractor, preserve_exact_technical_tokens
from personal_brain.schema import BrainSchema, ClosingConnection
from scripts.adapters.feishu_bridge import FeishuBrainBridge
from tests.reliability.test_reliability import FakeClient, config_for, options, payload


class CountingChat:
    def __init__(self, response: str, *, block: bool = False):
        self.response = response
        self.calls = 0
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block

    @property
    def available(self):
        return True

    def chat(self, *args, **kwargs):
        with self.lock:
            self.calls += 1
        self.entered.set()
        if self.block:
            self.release.wait(5)
        return self.response


class ReadOnlyEmbedding:
    available = True

    def embed(self, text):
        return [1.0, 0.0]


REMEMBER_PAYLOAD = json.dumps({
    "should_remember": True,
    "atomic_memories": [{
        "title": "合成记忆", "content": "合成的长期记忆内容", "memory_category": "其他",
        "memory_type": "fact", "importance": 0.7, "confidence": 0.8,
        "topics": [], "entities": [],
    }],
}, ensure_ascii=False)


def seed_terminal_crash(brain: PersonalBrain, message_id: str, status: str, should_remember: bool, memory: bool):
    interaction_id, _ = brain.claim_interaction(
        message_id=message_id, source="feishu", sender="u", user_text="fixture", mode="remember"
    )
    brain.claim_interaction_processing(interaction_id)
    with brain.schema.connect_write() as conn:
        raw_cursor = conn.execute(
            """
            INSERT INTO raw_messages(content,source,sender,processed_status,source_message_id,processed_at)
            VALUES('fixture','feishu','u',?,?,datetime('now'))
            """,
            (status, message_id),
        )
        raw_id = int(raw_cursor.lastrowid)
        run_cursor = conn.execute(
            """
            INSERT INTO memory_extraction_runs(
              raw_message_id,model_provider,model_name,prompt_version,input_hash,output_json,status
            ) VALUES(?,?,?,?,?,?, 'succeeded')
            """,
            (raw_id, "fake", "fake", "fixture", "hash", json.dumps({
                "should_remember": should_remember, "atomic_memories": []
            })),
        )
        run_id = int(run_cursor.lastrowid)
        memory_id = None
        if memory:
            cursor = conn.execute(
                """
                INSERT INTO memories(raw_message_id,extraction_run_id,content,title,memory_category,
                  memory_type,importance,confidence) VALUES(?,?,?,?,?,?,?,?)
                """,
                (raw_id, run_id, "已提交的记忆", "恢复记忆", "其他", "fact", .7, .8),
            )
            memory_id = int(cursor.lastrowid)
        conn.execute(
            "UPDATE interaction_logs SET updated_at=datetime('now','-20 minutes') WHERE id=?",
            (interaction_id,),
        )
    return interaction_id, raw_id, run_id, memory_id


def counts(brain: PersonalBrain):
    with brain.schema.connect_readonly() as conn:
        return tuple(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in (
            "raw_messages", "memory_extraction_runs", "memories"
        ))


class ReliabilityR2Tests(unittest.TestCase):
    def test_r1_v2_history_maps_terminal_and_bridge_replays_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db = root / "v2.sqlite3"
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                conn.executescript(
                    """
                    CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
                    INSERT INTO schema_migrations VALUES(2,'old');
                    CREATE TABLE interaction_logs(
                      id INTEGER PRIMARY KEY, message_id TEXT, source TEXT NOT NULL, sender TEXT NOT NULL,
                      user_text TEXT NOT NULL, mode TEXT NOT NULL, action TEXT NOT NULL, raw_message_id INTEGER,
                      reply_text TEXT, evidence_json TEXT, status TEXT NOT NULL, error TEXT, latency_ms INTEGER,
                      created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                    INSERT INTO interaction_logs(id,message_id,source,sender,user_text,mode,action,reply_text,status)
                      VALUES(1,'a','feishu','u','x','remember','remember','old reply','succeeded');
                    INSERT INTO interaction_logs(id,message_id,source,sender,user_text,mode,action,status)
                      VALUES(2,'b','feishu','u','x','remember','error','failed');
                    INSERT INTO interaction_logs(id,message_id,source,sender,user_text,mode,action,status)
                      VALUES(3,'c','feishu','u','x','remember','stale_ignored','succeeded');
                    INSERT INTO interaction_logs(id,message_id,source,sender,user_text,mode,action,status)
                      VALUES(4,'d','feishu','u','x','remember','remember','custom_old_state');
                    """
                )
            schema = BrainSchema(db)
            schema.initialize()
            with schema.connect_readonly() as conn:
                rows = conn.execute(
                    "SELECT id,status,processing_status,delivery_status,idempotency_key,attempt_count,delivered_at "
                    "FROM interaction_logs ORDER BY id"
                ).fetchall()
            self.assertEqual(
                [(r["status"], r["processing_status"], r["delivery_status"]) for r in rows],
                [("succeeded","succeeded","unknown"), ("failed","failed","unknown"),
                 ("ignored","ignored","not_required"), ("custom_old_state","legacy_terminal","unknown")],
            )
            self.assertTrue(all(r["idempotency_key"] is None and r["attempt_count"] == 0 and r["delivered_at"] is None for r in rows))
            brain = PersonalBrain(BrainConfig(db, root / "router", root / "index.json"))
            self.assertEqual(brain.recoverable_interactions(), [])
            client = FakeClient()
            bridge = FeishuBrainBridge(brain, client, options())
            bridge._jobs.join()
            self.assertEqual(client.replies, [])
            bridge.shutdown()

    def test_r1_mapping_failure_rolls_back_all_history(self):
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "v2.sqlite3"
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                conn.executescript(
                    """
                    CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT);
                    INSERT INTO schema_migrations VALUES(2,'old');
                    CREATE TABLE interaction_logs(id INTEGER PRIMARY KEY, message_id TEXT, source TEXT NOT NULL,
                      sender TEXT NOT NULL,user_text TEXT NOT NULL,mode TEXT NOT NULL,action TEXT NOT NULL,
                      raw_message_id INTEGER,reply_text TEXT,evidence_json TEXT,status TEXT NOT NULL,error TEXT,
                      latency_ms INTEGER,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
                    INSERT INTO interaction_logs(id,source,sender,user_text,mode,action,status)
                      VALUES(1,'feishu','u','x','remember','remember','succeeded');
                    """
                )
            with self.assertRaisesRegex(RuntimeError, "injected migration failure"):
                BrainSchema(db).initialize(fail_at="after_additive_migration")
            with sqlite3.connect(db, factory=ClosingConnection) as conn:
                self.assertEqual(conn.execute("SELECT status FROM interaction_logs WHERE id=1").fetchone()[0], "succeeded")
                self.assertNotIn("processing_status", {r[1] for r in conn.execute("PRAGMA table_info(interaction_logs)")})

    def _run_terminal(self, status: str, should_remember: bool, memory: bool, fail_delivery=False):
        td = tempfile.TemporaryDirectory()
        root = Path(td.name)
        brain = PersonalBrain(config_for(root))
        brain.init_db()
        model = CountingChat(REMEMBER_PAYLOAD)
        brain.chat_model = model
        interaction_id, raw_id, run_id, memory_id = seed_terminal_crash(
            brain, f"case-{status}-{memory}", status, should_remember, memory
        )
        before = counts(brain)
        client = FakeClient(fail=fail_delivery)
        bridge = FeishuBrainBridge(brain, client, options())
        bridge._jobs.join()
        after = counts(brain)
        self.assertEqual(after, before)
        self.assertEqual(model.calls, 0)
        return td, brain, bridge, client, interaction_id, raw_id, memory_id

    def test_r2_p_processed_memory_resumes_without_models(self):
        td, brain, bridge, client, iid, raw_id, memory_id = self._run_terminal("processed", True, True)
        row = brain.get_interaction(iid)
        self.assertEqual((row["processing_status"], row["delivery_status"], row["status"]), ("succeeded","succeeded","succeeded"))
        self.assertEqual(row["raw_message_id"], raw_id)
        self.assertIn(f"记忆ID：{memory_id}", row["reply_text"])
        self.assertNotIn("暂时处理失败", row["reply_text"])
        bridge.shutdown(); td.cleanup()

    def test_r2_p0_processed_zero_memory_normalized(self):
        td, brain, bridge, client, iid, raw_id, _ = self._run_terminal("processed", True, False)
        row = brain.get_interaction(iid)
        self.assertEqual(row["action"], "received")
        self.assertIn("没有新增长期记忆", row["reply_text"])
        self.assertNotIn("暂时处理失败", row["reply_text"])
        bridge.shutdown(); td.cleanup()

    def test_r2_i_ignored_normalized_and_truthful(self):
        td, brain, bridge, client, iid, raw_id, _ = self._run_terminal("ignored", False, False)
        row = brain.get_interaction(iid)
        self.assertEqual((row["action"], row["processing_status"], row["status"]), ("ignored","ignored","ignored"))
        self.assertIn("没有写入长期记忆", row["reply_text"])
        bridge.shutdown(); td.cleanup()

    def test_r2_x_inconsistent_terminal_fails_closed_without_model(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            model = CountingChat(REMEMBER_PAYLOAD)
            brain.chat_model = model
            iid, _ = brain.claim_interaction(message_id="x", source="feishu", sender="u", user_text="fixture", mode="remember")
            brain.claim_interaction_processing(iid)
            with brain.schema.connect_write() as conn:
                conn.execute("INSERT INTO raw_messages(content,source,sender,processed_status,source_message_id) VALUES('x','feishu','u','processed','x')")
                conn.execute("UPDATE interaction_logs SET updated_at=datetime('now','-20 minutes') WHERE id=?", (iid,))
            before = counts(brain)
            client = FakeClient()
            bridge = FeishuBrainBridge(brain, client, options())
            bridge._jobs.join()
            row = brain.get_interaction(iid)
            self.assertEqual(row["processing_status"], "failed")
            self.assertIn("committed recovery evidence is inconsistent", row["error"])
            self.assertEqual(counts(brain), before)
            self.assertEqual(model.calls, 0)
            self.assertNotIn("已记住", client.replies[0][1])
            bridge.shutdown()
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            model = CountingChat(REMEMBER_PAYLOAD)
            brain.chat_model = model
            iid, _, _, _ = seed_terminal_crash(brain, "x-mismatch", "ignored", True, False)
            before = counts(brain)
            bridge = FeishuBrainBridge(brain, FakeClient(), options())
            bridge._jobs.join()
            row = brain.get_interaction(iid)
            self.assertEqual(row["processing_status"], "failed")
            self.assertIn("raw/run terminal mismatch", row["error"])
            self.assertEqual(counts(brain), before)
            self.assertEqual(model.calls, 0)
            bridge.shutdown()

    def test_r2_d_delivery_retry_only_reuses_saved_reply(self):
        td, brain, bridge, client, iid, raw_id, memory_id = self._run_terminal("processed", True, True, fail_delivery=True)
        first = brain.get_interaction(iid)
        self.assertEqual(first["delivery_status"], "failed")
        saved = first["reply_text"]
        bridge.shutdown()
        before = counts(brain)
        retry_client = FakeClient()
        retry = FeishuBrainBridge(brain, retry_client, options())
        retry._jobs.join()
        self.assertEqual(retry_client.replies, [(first["message_id"], saved)])
        self.assertEqual(counts(brain), before)
        self.assertEqual(brain.chat_model.calls, 0)
        retry.shutdown(); td.cleanup()

    def test_r3_business_methods_never_initialize_and_old_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            brain = PersonalBrain(config_for(root))
            brain.init_db()
            brain.chat_model = CountingChat(REMEMBER_PAYLOAD)
            with patch.object(brain.schema, "initialize", side_effect=AssertionError("unexpected initialize")):
                result = brain.ingest("fixture", rebuild_router=False)
                brain.record_interaction(message_id=None, source="test", sender="u", user_text="x", mode="x", action="x", status="succeeded")
                brain.archive_memory(result.memory_ids[0], rebuild_router=False)
                brain.embed_missing_memories(limit=1)
                with patch("personal_brain.vault.dpapi_protect", return_value=b"synthetic"):
                    brain.secure_add("fixture", "note", "secret", "master")
            missing = PersonalBrain(config_for(root / "missing"))
            with self.assertRaisesRegex(FileNotFoundError, "run init-db"):
                missing.ingest("x", rebuild_router=False)
            self.assertFalse((root / "missing" / "data" / "test.sqlite3").exists())

    def test_r4_end_to_end_twenty_webhooks_create_one_everything(self):
        with tempfile.TemporaryDirectory() as td:
            brain = PersonalBrain(config_for(Path(td)))
            brain.init_db()
            model = CountingChat(REMEMBER_PAYLOAD)
            brain.chat_model = model
            client = FakeClient()
            bridge = FeishuBrainBridge(brain, client, options())
            with ThreadPoolExecutor(max_workers=20) as pool:
                results = list(pool.map(lambda i: bridge.handle_payload(payload("e2e", f"e2e-{i}", "fixture")), range(20)))
            bridge._jobs.join()
            self.assertEqual(sum("accepted" in item for item in results), 1)
            self.assertEqual(model.calls, 1)
            self.assertEqual(counts(brain), (1, 1, 1))
            self.assertEqual(len(client.replies), 1)
            self.assertEqual(len(brain.list_interactions(50)), 1)
            bridge.shutdown()

    def test_r4_concurrent_reprocess_has_one_claim_and_model_call(self):
        with tempfile.TemporaryDirectory() as td:
            schema = BrainSchema(Path(td) / "brain.sqlite3")
            schema.initialize()
            model = CountingChat(json.dumps({"should_remember": False, "atomic_memories": []}), block=True)
            extractor = MemoryExtractor(schema, model, ChatModelConfig())
            raw_id, _ = extractor.capture_raw("x", "test", "u")
            with schema.connect_write() as conn:
                conn.execute("UPDATE raw_messages SET processed_status='failed' WHERE id=?", (raw_id,))
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(extractor.reprocess, raw_id)
                self.assertTrue(model.entered.wait(3))
                second = pool.submit(extractor.reprocess, raw_id)
                time.sleep(.1)
                model.release.set()
                first.result()
                with self.assertRaises(RuntimeError):
                    second.result()
            self.assertEqual(model.calls, 1)

    def test_r5_interaction_list_cli_shows_split_states(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = config_for(root)
            BrainSchema(config.database_path).initialize()
            brain = PersonalBrain(config)
            brain.claim_interaction(message_id="cli", source="feishu", sender="u", user_text="x", mode="remember")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "database_path": str(config.database_path), "memory_dir": str(config.memory_dir),
                "brain_index_path": str(config.brain_index_path)
            }), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                code = brain_cli.main(["--config", str(config_path), "interaction-list", "--limit", "1"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("processing=pending", text)
            self.assertIn("delivery=pending", text)
            self.assertIn("attempt=0", text)

    def test_r4_readonly_matrix_existing_db_does_not_mutate_store(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            brain = PersonalBrain(config_for(root))
            brain.init_db()
            brain.chat_model = CountingChat(json.dumps({
                "summary": "ok", "overall_score": 1, "strengths": [], "problems": [],
                "recommendations": [], "merge_groups": [], "prompt_improvements": []
            }))
            brain.embedding_model = ReadOnlyEmbedding()
            brain.semantic_memory.embedding_client = brain.embedding_model
            brain.answer_engine.chat_model = brain.chat_model
            before_bytes = brain.config.database_path.read_bytes()
            before_stat = brain.config.database_path.stat()
            before_files = sorted(p.name for p in brain.config.database_path.parent.iterdir())
            with brain.schema.connect_readonly() as conn:
                before_version = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
                before_journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
            brain.stats()
            self.assertEqual(brain.memory_list(), [])
            with self.assertRaises(KeyError): brain.memory_show(999)
            self.assertEqual(brain.list_interactions(), [])
            brain.daily_report(__import__('datetime').date.today(), root / "reports")
            brain.review_memories(limit=1)
            self.assertEqual(brain.recall("x"), [])
            self.assertEqual(brain.ask("x").evidence, [])
            self.assertEqual(brain.secure_list(), [])
            with self.assertRaises(KeyError): brain.secure_get("missing", "master")
            brain.build_router()
            with brain.schema.connect_readonly() as conn:
                self.assertEqual(conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0], before_version)
                self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0], before_journal)
            self.assertEqual(brain.config.database_path.read_bytes(), before_bytes)
            after_stat = brain.config.database_path.stat()
            self.assertEqual((after_stat.st_mtime_ns, after_stat.st_size), (before_stat.st_mtime_ns, before_stat.st_size))
            self.assertEqual(sorted(p.name for p in brain.config.database_path.parent.iterdir()), before_files)

    def test_r4_readonly_matrix_missing_db_creates_no_store_or_router(self):
        operations = (
            lambda b, r: b.stats(), lambda b, r: b.memory_list(), lambda b, r: b.memory_show(1),
            lambda b, r: b.list_interactions(),
            lambda b, r: b.daily_report(__import__('datetime').date.today(), r / "reports"),
            lambda b, r: b.review_memories(limit=1), lambda b, r: b.recall("x"),
            lambda b, r: b.ask("x"), lambda b, r: b.secure_list(),
            lambda b, r: b.secure_get("x", "master"), lambda b, r: b.build_router(),
        )
        for index, operation in enumerate(operations):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                brain = PersonalBrain(config_for(root))
                brain.chat_model = CountingChat("{}")
                brain.embedding_model = ReadOnlyEmbedding()
                brain.semantic_memory.embedding_client = brain.embedding_model
                brain.answer_engine.chat_model = brain.chat_model
                with self.assertRaises((FileNotFoundError, RuntimeError, KeyError)):
                    operation(brain, root)
                self.assertFalse(brain.config.database_path.exists())
                self.assertFalse(brain.config.database_path.parent.exists())
                self.assertFalse(brain.config.memory_dir.exists())
                self.assertFalse(brain.config.brain_index_path.exists())

    def test_extractor_v7_exact_token_guard_regression(self):
        payload_data = {"should_remember": True, "atomic_memories": [{"content": "调用 .loads 和 .dumps，写入 memory.", "title": "x"}]}
        fixed = preserve_exact_technical_tokens("使用 json.loads 和 json.dumps 写入 memory.json", payload_data)
        text = json.dumps(fixed, ensure_ascii=False)
        self.assertIn("json.loads", text)
        self.assertIn("json.dumps", text)
        self.assertIn("memory.json", text)


if __name__ == "__main__":
    unittest.main()

