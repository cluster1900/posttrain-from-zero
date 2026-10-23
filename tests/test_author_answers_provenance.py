"""Bind author-answer numbers to data/*.json; mutations must fail re-bind."""
from __future__ import annotations

import json
import random
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "docs" / "author-answers.md"
Q4 = ROOT / "data" / "q4_lora_2b_parameters.json"
Q6 = ROOT / "data" / "q6_paired_bootstrap.json"
Q79 = ROOT / "data" / "q7_q9_rl_metrics.json"
Q8 = ROOT / "data" / "q8_entropy_format.json"


def paired_bootstrap(cur: dict[str, float], base: dict[str, float], iters: int = 2000, seed: int = 0):
    tasks = sorted(set(cur) & set(base))
    diffs = [cur[t] - base[t] for t in tasks]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
    return sum(diffs) / len(diffs), means[int(0.025 * iters)], means[int(0.975 * iters)], len(tasks)


class AuthorAnswerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.text = ANSWERS.read_text(encoding="utf-8")
        self.q4 = json.loads(Q4.read_text(encoding="utf-8"))
        self.q6 = json.loads(Q6.read_text(encoding="utf-8"))
        self.q79 = json.loads(Q79.read_text(encoding="utf-8"))
        self.q8 = json.loads(Q8.read_text(encoding="utf-8"))

    def _question(self, number: int) -> str:
        start = self.text.index(f"## Q{number}：")
        end = self.text.find("\n## Q", start + 1)
        if end < 0:
            end = self.text.find("\n## 复现", start + 1)
        return self.text[start:] if end < 0 else self.text[start:end]

    def test_a8_structure_and_topic_specific_links(self):
        self.assertNotIn("qwen" + "-posttrain", self.text.lower())
        for q in range(1, 11):
            with self.subTest(question=q):
                chunk = self._question(q)
                for heading in ("### 记录依据", "### 常见做法与出处", "### 结论"):
                    self.assertIn(heading, chunk)
                common = chunk.split("### 常见做法与出处", 1)[1].split("\n### ", 1)[0]
                self.assertRegex(common, r"https?://")
                conclusion = chunk.split("### 结论", 1)[1]
                first = next(line for line in conclusion.splitlines() if line.strip())
                self.assertRegex(first, r"【(记录事实|读者运行所得|作者事后分析)】")
        for q, needle in {
            6: "arxiv.org/abs/2107.03374",
            7: "arxiv.org/abs/1707.06347",
            8: "arxiv.org/abs/1707.06347",
            9: "arxiv.org/abs/2107.03374",
            10: "huggingface.co/Qwen/Qwen3.5-2B-Base",
        }.items():
            self.assertIn(needle, self._question(q))

    def _assert_q4_bound(self, text: str, data: dict) -> None:
        claimed = {}
        for key in (
            "base_unique_parameters",
            "lora_trainable_parameters",
            "wrapped_unique_parameters",
            "n_target_instances",
        ):
            match = re.search(rf"{key}=(\d+)", text)
            self.assertIsNotNone(match, key)
            claimed[key] = int(match.group(1))
            self.assertEqual(claimed[key], data[key], key)
        self.assertIn(data["revision"], text)
        frac = re.search(r"0\.0345168", text)
        self.assertIsNotNone(frac)
        self.assertAlmostEqual(float(frac.group()), data["trainable_fraction"], places=7)

    def test_q4_numbers_bind_to_json(self):
        self._assert_q4_bound(self._question(4), self.q4)
        self.assertEqual(self.q4["lora_trainable_parameters"], 67276800)
        self.assertEqual(self.q4["n_target_instances"], 186)

    def test_q4_mutation_of_json_or_text_fails_rebind(self):
        corrupted = dict(self.q4)
        corrupted["lora_trainable_parameters"] = 1
        with self.assertRaises(AssertionError):
            self._assert_q4_bound(self._question(4), corrupted)
        mutated_text = self._question(4).replace(
            "lora_trainable_parameters=67276800",
            "lora_trainable_parameters=67276801",
            1,
        )
        with self.assertRaises(AssertionError):
            self._assert_q4_bound(mutated_text, self.q4)

    def _assert_q6_bound(self, text: str, data: dict) -> None:
        sft = data["paired_bootstrap"]["sft_minus_baseline"]
        rl = data["paired_bootstrap"]["rl_minus_sft"]
        # Text must quote the JSON literals; then recompute from per-task tables.
        for value in (
            sft["mean_delta"],
            sft["ci_low"],
            sft["ci_high"],
            rl["mean_delta"],
            rl["ci_low"],
            rl["ci_high"],
        ):
            self.assertIn(str(value), text)
        mean, lo, hi, n = paired_bootstrap(
            data["per_task_pass_rate"]["sft"],
            data["per_task_pass_rate"]["baseline"],
            iters=sft["iters"],
            seed=sft["seed"],
        )
        self.assertEqual(n, sft["n_tasks"])
        self.assertEqual(mean, sft["mean_delta"])
        self.assertEqual(lo, sft["ci_low"])
        self.assertEqual(hi, sft["ci_high"])
        # Parse the SFT mean quoted in the conclusion and require equality.
        match = re.search(r"SFT−Base 均值增益 `([^`]+)`", text)
        self.assertIsNotNone(match)
        self.assertEqual(float(match.group(1)), sft["mean_delta"])

    def test_q6_bootstrap_binds_and_recomputes(self):
        self._assert_q6_bound(self._question(6), self.q6)

    def test_q6_mutation_fails_rebind(self):
        corrupted = json.loads(json.dumps(self.q6))
        corrupted["paired_bootstrap"]["sft_minus_baseline"]["mean_delta"] = 0.5
        with self.assertRaises(AssertionError):
            self._assert_q6_bound(self._question(6), corrupted)
        original = str(self.q6["paired_bootstrap"]["sft_minus_baseline"]["mean_delta"])
        mutated = self._question(6).replace(
            f"SFT−Base 均值增益 `{original}`",
            "SFT−Base 均值增益 `0.5`",
            1,
        )
        self.assertIn("SFT−Base 均值增益 `0.5`", mutated)
        with self.assertRaises(AssertionError):
            self._assert_q6_bound(mutated, self.q6)

    def _assert_q9_bound(self, text: str, data: dict) -> None:
        display = data["display"]
        for value in display["step_fracs_rounded"]:
            self.assertIn(str(value), text)
        match = re.search(r"平均 `([0-9.]+)`", text)
        self.assertIsNotNone(match)
        claimed = float(match.group(1))
        self.assertEqual(claimed, display["average_rounded"])
        avg = sum(s["frac"] for s in data["steps"]) / len(data["steps"])
        self.assertAlmostEqual(avg, display["average_rounded"], places=7)
        self.assertAlmostEqual(avg, data["recent_no_tool_call_frac"], places=12)

    def test_q7_q9_numbers_bind(self):
        q7 = self._question(7)
        self.assertIn(str(self.q79["kl_at_step_5"]), q7)
        self.assertIn(str(self.q79["display"]["kl_step5_full"]), q7)
        self._assert_q9_bound(self._question(9), self.q79)

    def test_q9_mutation_fails_rebind(self):
        mutated = self._question(9).replace("平均 `0.0822917`", "平均 `0.9999999`", 1)
        with self.assertRaises(AssertionError):
            self._assert_q9_bound(mutated, self.q79)
        corrupted = json.loads(json.dumps(self.q79))
        corrupted["display"]["average_rounded"] = 0.9999999
        with self.assertRaises(AssertionError):
            self._assert_q9_bound(self._question(9), corrupted)

    def test_q8_entropy_and_format_gates_bind(self):
        q8 = self._question(8)
        self.assertIn(str(self.q8["entropy_first"]), q8)
        self.assertIn(str(self.q8["entropy_last"]), q8)
        self.assertIn(str(self.q8["recent_no_tool_call_frac"]), q8)
        self.assertIn("0.10", q8)
        ratio = self.q8["entropy_last"] / self.q8["entropy_first"]
        self.assertLess(ratio, self.q8["gates"]["entropy_blowup_factor"])
        self.assertLessEqual(self.q8["recent_no_tool_call_frac"], self.q8["gates"]["no_tool_call_max"])


if __name__ == "__main__":
    unittest.main()
