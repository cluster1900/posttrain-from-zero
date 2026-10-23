import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ch09_rl_2b_24k_metrics.json"
CHAPTER = ROOT / "chapters" / "09_dppo.md"


class Ch09EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(DATA.read_text(encoding="utf-8"))
        self.chapter = CHAPTER.read_text(encoding="utf-8")

    def test_evidence_has_real_curves_and_source(self):
        rows = self.data["records"]
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(self.data["source"]["commit"], "ca57456")
        self.assertIn("DATA_SOURCE", (ROOT / "scripts/figures/ch09_rl_2b_24k.py").read_text(encoding="utf-8"))
        provenance = (ROOT / "docs" / "ca57456-provenance.md").read_text(encoding="utf-8")
        for marker in ("ca57456", "result/round2/results/rl_2b_24k_metrics.jsonl", "ch09_time_split.svg"):
            self.assertIn(marker, provenance)
        for name in ("ch09_loop.svg", "ch09_filters.svg", "ch09_mask.svg", "ch09_memory.svg"):
            svg = (ROOT / "assets" / "images" / name).read_text(encoding="utf-8")
            self.assertIn("Noto Sans CJK SC", svg)
        for row in rows:
            with self.subTest(step=row.get("step")):
                self.assertGreaterEqual(row["reward_mean"], 0)
                self.assertLessEqual(row["reward_mean"], 1)
                self.assertGreaterEqual(row["kept_frac"], 0)
                self.assertLessEqual(row["kept_frac"], 1)
                self.assertLessEqual(row["useful_groups"], row["groups"])
                for key in ("gen_s", "train_s", "step_s", "sync_logprob_abs_diff"):
                    self.assertGreaterEqual(row[key], 0)

    def test_chapter_numbers_are_bound_to_data(self):
        for row in self.data["records"]:
            self.assertIn(str(row["step_s"]), self.chapter)
            self.assertIn(f"{row['sync_logprob_abs_diff']:.10f}", self.chapter)

    def test_mutating_a_bound_number_breaks_the_contract(self):
        row = self.data["records"][0]
        original = str(row["step_s"])
        mutated = self.chapter.replace(original, "999.9", 1)
        self.assertNotEqual(mutated, self.chapter)
        self.assertNotIn(original, mutated)
        self.assertNotIn("999.9", "\n".join(str(r["step_s"]) for r in self.data["records"]))


if __name__ == "__main__":
    unittest.main()
