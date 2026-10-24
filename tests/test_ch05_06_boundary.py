import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEG = re.compile(r'不代表|不等于|不能|不是|并不|不保证|不说明|不意味着|并非|不构成|不表示')
# Caveats that must survive the compression, one anchor per true limitation.
CAVEATS = {
    '05': ['不执行命令', 'UTF-8 字节层', '逐字替代源 `parse_assistant_response`', '按字符计',
           '全局 eos', '`c33ee0d`'],
    '06': ['hidden=64', '143,872', '75%', 'forward 正确性', '不回填', '不称其最优', '不代表作者当时'],
}


def chapter(n):
    return next((ROOT / 'chapters').glob(f'{n}_*.md')).read_text(encoding='utf-8')


def split(text):
    m = re.search(r'^## 边界说明\n(.*?)(?=^## )', text, re.M | re.S)
    return m.group(1), text[:m.start()] + text[m.end():]


class BoundarySectionTests(unittest.TestCase):
    def test_single_boundary_section_holds_the_caveats(self):
        for n, anchors in CAVEATS.items():
            text = chapter(n)
            self.assertEqual(text.count('\n## 边界说明\n'), 1, n)
            box, _ = split(text)
            for anchor in anchors:
                self.assertIn(anchor, box, (n, anchor))

    def test_body_outside_boundary_section_is_not_padded_with_negations(self):
        for n in CAVEATS:
            _, body = split(chapter(n))
            # headings, tables of pitfalls and the source list may keep a few factual negations
            self.assertLessEqual(len(NEG.findall(body)), 3, (n, NEG.findall(body)))


if __name__ == '__main__':
    unittest.main()
