"""Bind chapter 10 probe numbers to data/ch10_sandbox_probe.json."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ch10_sandbox_probe.json"
CHAPTER = ROOT / "chapters" / "10_sandbox.md"
SCRIPT = ROOT / "scripts" / "figures" / "ch10_sandbox_probe.py"


class Ch10SandboxProbeTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.chapter = CHAPTER.read_text(encoding="utf-8")

    def _assert_bound(self, chapter: str, data: dict) -> None:
        match = re.search(r"沙箱创建约 ([0-9.]+) 秒", chapter)
        self.assertIsNotNone(match)
        self.assertEqual(float(match.group(1)), float(data["sandbox_create_s"]))
        for cmd in data["commands"]:
            token = f"{cmd['label']}={cmd['elapsed_s']}s"
            self.assertIn(token, chapter)
        self.assertIn("](images/plots/ch10_sandbox_probe.svg)", chapter)
        self.assertIn("FAILED_PRECONDITION", chapter)
        self.assertIn("DEADLINE_EXCEEDED", chapter)
        self.assertIn(str(data["probe_process_exit_code"]), chapter)
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("DATA_SOURCE", script)
        self.assertIn("--out", script)

    def test_chapter_binds_to_probe_json(self):
        self.assertEqual(self.data["source"]["commit"], "ca57456")
        self._assert_bound(self.chapter, self.data)

    def test_mutation_fails_rebind(self):
        corrupted = json.loads(json.dumps(self.data))
        corrupted["sandbox_create_s"] = 0.1
        with self.assertRaises(AssertionError):
            self._assert_bound(self.chapter, corrupted)
        mutated = self.chapter.replace(
            f"沙箱创建约 {self.data['sandbox_create_s']} 秒",
            "沙箱创建约 0.1 秒",
            1,
        )
        with self.assertRaises(AssertionError):
            self._assert_bound(mutated, self.data)


if __name__ == "__main__":
    unittest.main()
