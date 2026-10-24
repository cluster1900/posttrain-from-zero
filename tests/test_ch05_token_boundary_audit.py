import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import ch05_token_boundary_audit as A  # noqa: E402


def toy_encode(s):
    """Greedy toy BPE: '\\n\\n' merges into one token (271), everything else per char."""
    out, i = [], 0
    while i < len(s):
        if s.startswith("\n\n", i):
            out.append(271); i += 2
        else:
            out.append(ord(s[i])); i += 1
    return out


def toy_offsets(s):
    out, i = [], 0
    while i < len(s):
        n = 2 if s.startswith("\n\n", i) else 1
        out.append((i, i + n)); i += n
    return out


class AuditTest(unittest.TestCase):
    text = "<think>\n\n</think>"
    cut = len("<think>\n")  # generation prefix ends between the two newlines

    def test_piecewise_splits_merged_token(self):
        whole = toy_encode(self.text)
        piece = A.piecewise_ids(toy_encode, self.text, [self.cut])
        self.assertIn(271, whole)
        self.assertNotIn(271, piece)
        self.assertEqual(len(piece) - len(whole), 1)
        self.assertEqual(A.crossed_boundaries(toy_offsets(self.text), [self.cut]), [self.cut])

    def test_cut_on_token_edge_changes_nothing(self):
        cut = len("<think>")
        self.assertEqual(A.piecewise_ids(toy_encode, self.text, [cut]), toy_encode(self.text))
        self.assertEqual(A.crossed_boundaries(toy_offsets(self.text), [cut]), [])
        # out-of-range / duplicate cuts are ignored
        self.assertEqual(A.piecewise_ids(toy_encode, self.text, [0, cut, cut, 999]), toy_encode(self.text))

    def test_summarize(self):
        recs = [
            dict(reason="ok", piece_ids_equal_whole=False, assistant_turns=3, len_piece=11, len_whole=10,
                 boundaries=6, crossed=1, crossed_start=1, label_tokens=4, stats={"a": 1}),
            dict(reason="ok", piece_ids_equal_whole=True, assistant_turns=3, len_piece=20000, len_whole=20000,
                 boundaries=6, crossed=0, crossed_start=0, label_tokens=7, stats={"a": 2, "b": 1}),
        ]
        s = A.summarize(recs)
        self.assertEqual((s["trajectories"], s["piece_equal_whole"], s["piece_differs_from_whole"]), (2, 1, 1))
        self.assertEqual((s["span_boundaries"], s["crossed_boundaries"]), (12, 1))
        self.assertTrue(s["delta_equals_crossed_for_all"])
        self.assertEqual(s["retained_at"], {"16384": 1, "24576": 2, "32768": 2})
        self.assertEqual(s["turn_stats"], {"a": 3, "b": 1})
        recs[0]["crossed"] = 0  # mutation: delta no longer explained
        self.assertFalse(A.summarize(recs)["delta_equals_crossed_for_all"])


if __name__ == "__main__":
    unittest.main()
