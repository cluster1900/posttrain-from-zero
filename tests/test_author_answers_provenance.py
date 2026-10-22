from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = ROOT / "docs" / "author-answers.md"


class AuthorAnswerProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.text = ANSWERS.read_text(encoding="utf-8")

    def test_author_answers_have_a8_sections_and_source_labels(self):
        self.assertIn("[记录事实]", self.text)
        self.assertIn("[读者运行所得]", self.text)
        self.assertIn("[作者分析]", self.text)
        self.assertIn("qwen-posttrain@ca57456", self.text)
        self.assertIn("私有 URL、账号、主机名、云项目或凭据", self.text)
        for q in range(1, 11):
            with self.subTest(question=q):
                start = self.text.index(f"## Q{q}：")
                end = self.text.find("\n## Q", start + 1)
                chunk = self.text[start:] if end < 0 else self.text[start:end]
                for heading in ("### 记录依据", "### 常见做法与出处", "### 结论"):
                    self.assertIn(heading, chunk)
                self.assertRegex(chunk, r"https?://")

    def test_q9_window_numbers_are_bound_to_text(self):
        for value in ("0.0625", "0.125", "0.20", "0.0416667", "0.0822917"):
            self.assertIn(value, self.text)
        mutated = self.text.replace("0.0822917", "0.9999999", 1)
        self.assertNotEqual(mutated, self.text)
        q9 = mutated.split("## Q9", 1)[-1].split("## Q", 1)[0]
        self.assertNotIn("0.0822917", q9)


if __name__ == "__main__":
    unittest.main()
