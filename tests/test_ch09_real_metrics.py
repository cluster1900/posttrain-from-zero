"""Bind chapter 9 numbers to data/ch09_rl_metrics.json; mutations must fail re-bind."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ch09_rl_metrics.json"
CHAPTER = ROOT / "chapters" / "09_dppo.md"
SCRIPT = ROOT / "scripts" / "figures" / "ch09_rl_metrics.py"


class Ch09RealMetricsTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.chapter = CHAPTER.read_text(encoding="utf-8")
        self.cont = self.data["runs"]["round10_continue_final"]["records"]
        self.rehe = self.data["runs"]["round10_rehearsal_2b"]["records"]

    def test_scripts_declare_data_source_and_honor_out(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("DATA_SOURCE", text)
        self.assertIn('add_argument("--out"', text)
        for name in (
            "ch09_reward_useful",
            "ch09_kept_frac",
            "ch09_sync_diff",
            "ch09_time_split",
            "ch09_rehearsal_time",
        ):
            wrapper = (ROOT / "scripts/figures" / f"{name}.py").read_text(encoding="utf-8")
            self.assertIn("DATA_SOURCE", wrapper)
            self.assertIn("--out", wrapper)
            svg = (ROOT / "assets/images/plots" / f"{name}.svg").read_text(encoding="utf-8")
            self.assertIn("Noto Sans CJK SC", svg)

    def _assert_bound(self, chapter: str, data: dict) -> None:
        cont = data["runs"]["round10_continue_final"]["records"]
        rehe = data["runs"]["round10_rehearsal_2b"]["records"]
        match = re.search(r"各步 `time/step_s`：([0-9., ]+)", chapter)
        self.assertIsNotNone(match)
        claimed = [float(x.strip()) for x in match.group(1).split(",")]
        self.assertEqual(claimed, [float(r["step_s"]) for r in cont])
        for row in cont:
            self.assertIn(f"{row['sync_logprob_abs_diff']:.10f}", chapter)
        match = re.search(r"三步 `time/step_s` 为 ([0-9.]+)、([0-9.]+)、([0-9.]+)", chapter)
        self.assertIsNotNone(match)
        self.assertEqual(
            [float(match.group(i)) for i in (1, 2, 3)],
            [float(r["step_s"]) for r in rehe],
        )
        for name in (
            "ch09_reward_useful.svg",
            "ch09_kept_frac.svg",
            "ch09_sync_diff.svg",
            "ch09_time_split.svg",
            "ch09_rehearsal_time.svg",
        ):
            self.assertIn(f"](images/plots/{name})", chapter)
        self.assertIn("1037af0", chapter)
        self.assertIn("train.max_steps=8", chapter)
        kl = next(r["kl_k3_vs_base"] for r in cont if r["step"] == 5)
        self.assertIn(str(kl), chapter)

    def test_chapter_binds_to_continue_final_and_rehearsal(self):
        self.assertGreaterEqual(len(self.cont), 8)
        self.assertEqual(len(self.rehe), 3)
        self.assertEqual(self.data["source"]["commit"], "ca57456")
        self._assert_bound(self.chapter, self.data)

    def test_mutation_of_json_or_chapter_fails_rebind(self):
        corrupted = json.loads(json.dumps(self.data))
        corrupted["runs"]["round10_continue_final"]["records"][0]["step_s"] = 1.0
        with self.assertRaises(AssertionError):
            self._assert_bound(self.chapter, corrupted)
        original = str(self.cont[0]["step_s"])
        mutated = self.chapter.replace(
            f"各步 `time/step_s`：{original},",
            "各步 `time/step_s`：1.0,",
            1,
        )
        with self.assertRaises(AssertionError):
            self._assert_bound(mutated, self.data)


if __name__ == "__main__":
    unittest.main()
