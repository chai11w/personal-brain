from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO = REPO_ROOT / "demo" / "offline_demo.py"


class OfflineDemoTests(unittest.TestCase):
    def run_demo(self, *query: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(DEMO), *query],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_default_query_returns_traceable_evidence(self) -> None:
        result = self.run_demo()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("memory=101 raw=501", result.stdout)
        self.assertIn("Top synthetic evidence:", result.stdout)

    def test_unrelated_query_returns_no_evidence(self) -> None:
        result = self.run_demo("How", "does", "Neptune", "orbit?")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("No matching synthetic evidence found.", result.stdout)
        self.assertNotIn("memory=", result.stdout)


if __name__ == "__main__":
    unittest.main()
