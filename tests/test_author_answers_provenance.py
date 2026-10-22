from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "docs" / "author-answers.md"


class AuthorAnswerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.text = ANSWERS.read_text(encoding="utf-8")

    def _question(self, number):
        start = self.text.index(f"## Q{number}：")
        end = self.text.find("\n## Q", start + 1)
        return self.text[start:] if end < 0 else self.text[start:end]

    def test_author_answers_have_a8_sections_and_public_sources(self):
        self.assertIn("[记录事实]", self.text)
        self.assertIn("【读者运行所得】", self.text)
        self.assertIn("【作者事后分析】", self.text)
        self.assertIn("固定实验提交 ca57456", self.text)
        self.assertNotIn("qwen-posttrain", self.text.lower())
        self.assertNotIn("私有 URL、账号、主机名、云项目或凭据", self.text)
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

    def test_q4_formula_and_q9_window_numbers_are_bound_to_text(self):
        q4 = self._question(4)
        self.assertIn("64×(4096+4096)=524288", q4)
        self.assertIn("1048576", q4)
        q9 = self._question(9)
        for value in ("0.0625", "0.125", "0.20", "0.0416667", "0.0822917"):
            self.assertIn(value, q9)
        mutated = q9.replace("0.0822917", "0.9999999", 1)
        self.assertNotEqual(mutated, q9)
        self.assertNotIn("0.0822917", mutated)


if __name__ == "__main__":
    unittest.main()
