from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.security.check_public_paths import denied


REPO = Path(__file__).resolve().parents[2]
SECURITY = REPO / "scripts" / "security"
TEST_TEMP = REPO / ".tmp_tests" / "security_tools"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


def run(*args: object, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def init_repo(root: Path) -> None:
    run("git", "init", cwd=root)
    run("git", "config", "user.email", "security-test@example.invalid", cwd=root)
    run("git", "config", "user.name", "Security Test", cwd=root)


class PublicPathGuardTests(unittest.TestCase):
    def test_private_regression_matrix(self) -> None:
        private = (
            ".agents/project_memory.md",
            ".agents/skills/leak.md",
            ".private_backups/repo.bundle",
            ".env",
            ".env.local",
            "config.json",
            "config.local.json",
            "config.private.json",
            "x.sqlite3-wal",
            "x.sqlite3-shm",
            "x.db-wal",
            "x.dump",
            "x.sql.gz",
            "nested/config.json",
            "nested/x.sqlite3",
        )
        self.assertTrue(all(denied(path) for path in private))
        self.assertFalse(denied(".env.example"))
        self.assertFalse(denied("config.example.json"))
        self.assertFalse(denied("personal_brain/brain.py"))

    def test_mixed_candidate_tree_fails(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP, ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            init_repo(root)
            (root / "safe.py").write_text("print('synthetic')\n", encoding="utf-8")
            (root / ".agents" / "skills").mkdir(parents=True)
            (root / ".agents" / "skills" / "leak.md").write_text("synthetic test", encoding="utf-8")
            run("git", "add", "-f", ".", cwd=root)
            result = run(sys.executable, SECURITY / "check_public_paths.py", "--root", root, cwd=REPO, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("DENY path=.agents/skills/leak.md", result.stdout)


class PrivateOutputTests(unittest.TestCase):
    def test_backup_restore_private_success_and_public_refusal(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP, ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            init_repo(root)
            (root / ".gitignore").write_text("private/\n", encoding="utf-8")
            source = root / "source.sqlite3"
            with sqlite3.connect(source) as conn:
                conn.execute("CREATE TABLE sample (value TEXT)")
                conn.execute("INSERT INTO sample VALUES ('synthetic')")

            rejected = run(
                sys.executable,
                SECURITY / "backup_sqlite.py",
                source,
                root / "public-output",
                cwd=root,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("non-ignored Git path", rejected.stderr)
            self.assertFalse((root / "public-output").exists())

            private = root / "private" / "backup"
            run(sys.executable, SECURITY / "backup_sqlite.py", source, private, cwd=root)
            backup = next(private.glob("*.sqlite3"))
            manifest = backup.with_suffix(backup.suffix + ".manifest.json")
            restore = root / "private" / "restore"
            run(
                sys.executable,
                SECURITY / "verify_sqlite_backup.py",
                backup,
                restore,
                "--manifest",
                manifest,
                cwd=root,
            )
            second = run(
                sys.executable,
                SECURITY / "verify_sqlite_backup.py",
                backup,
                restore,
                "--manifest",
                manifest,
                cwd=root,
                check=False,
            )
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)


class CaptureWorktreeTests(unittest.TestCase):
    def test_staged_unstaged_untracked_special_and_ignored_inventory(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP, ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            init_repo(root)
            (root / ".gitignore").write_text("private/\n", encoding="utf-8")
            (root / "staged.txt").write_text("base\n", encoding="utf-8")
            (root / "unstaged.txt").write_text("base\n", encoding="utf-8")
            run("git", "add", ".gitignore", "staged.txt", "unstaged.txt", cwd=root)
            run("git", "commit", "-m", "base", cwd=root)

            (root / "staged.txt").write_text("staged change\n", encoding="utf-8")
            run("git", "add", "staged.txt", cwd=root)
            (root / "unstaged.txt").write_text("unstaged change\n", encoding="utf-8")
            special = "special name [brackets] #1.txt"
            (root / special).write_text("untracked\n", encoding="utf-8")
            (root / "private").mkdir()
            (root / "private" / "ignored.txt").write_text("ignored synthetic\n", encoding="utf-8")

            destination = root / "private" / "snapshots"
            run(sys.executable, SECURITY / "capture_worktree.py", destination, "--root", root, cwd=root)
            snapshot = next(destination.glob("worktree-*"))
            self.assertIn(b"staged.txt", (snapshot / "staged.patch").read_bytes())
            self.assertIn(b"unstaged.txt", (snapshot / "unstaged.patch").read_bytes())
            with zipfile.ZipFile(snapshot / "changed-and-untracked.zip") as archive:
                self.assertIn(special, archive.namelist())
                self.assertNotIn("private/ignored.txt", archive.namelist())
            manifest = json.loads((snapshot / "git-state.json").read_text(encoding="utf-8"))
            self.assertIn("private/ignored.txt", manifest["ignored_inventory"])
            self.assertFalse(manifest["ignored_files_archived"])
            self.assertTrue(manifest["staged_patch_sha256"])
            self.assertTrue(manifest["unstaged_patch_sha256"])


if __name__ == "__main__":
    unittest.main()
