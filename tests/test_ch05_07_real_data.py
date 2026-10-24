import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fig", ROOT / "scripts/figures/ch05_07_real_data.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
STEMS = ("ch05_turns", "ch05_token_boundary", "ch06_lora_scale", "ch07_time_vs_tokens")


def chapter(n):
    return next((ROOT / "chapters").glob(f"{n}_*.md")).read_text()


class RealDataTest(unittest.TestCase):
    def test_ch05_json_consistent_and_in_chapter(self):
        d = json.loads((ROOT / "data/ch05_token_boundary.json").read_text())
        self.assertEqual(d["piece_equal_whole"] + d["piece_differs_from_whole"], d["trajectories"])
        self.assertEqual(sum(d["assistant_turns_hist"].values()), d["trajectories"])
        self.assertEqual(d["total_tokens_piecewise"] - d["total_tokens_whole"], d["crossed_boundaries"])
        self.assertEqual(sum(int(k) * v for k, v in d["extra_tokens_hist"].items()), d["crossed_boundaries"])
        self.assertEqual(d["span_boundaries"], 2 * sum(int(k) * v for k, v in d["assistant_turns_hist"].items()))
        self.assertTrue(d["delta_equals_crossed_for_all"])
        self.assertEqual(d["retained_at"], {"16384": 5330, "24576": 5618, "32768": 5711})
        t = chapter("05")
        for v in (d["trajectories"], d["piece_equal_whole"], d["piece_differs_from_whole"],
                  d["crossed_boundaries"], d["span_boundaries"], d["total_tokens_piecewise"]):
            self.assertIn(f"{v:,}", t)
        self.assertIn(f"{100 * d['piece_differs_from_whole'] / d['trajectories']:.1f}%", t)
        self.assertIn(f"{d['assistant_turns_hist']['64']} 条", t)

    def test_ch06_budget_matches_chapter(self):
        m = json.loads((ROOT / "data/ch06_lora_scale.json").read_text())["models"]
        t = chapter("06")
        for size in ("2b", "4b"):
            b = F.lora_budget(m[size])
            for k in ("base_bf16", "head_fp32", "lora_fp32_state", "static"):
                self.assertIn(f"{b[k]:.2f}", t, (size, k))
            self.assertIn(f"{m[size]['lora_trainable_parameters']:,}", t)
            self.assertIn(f"{m[size]['base_unique_parameters']:,}", t)
        self.assertAlmostEqual(F.lora_budget(m["2b"])["head_fp32"], 1.8945, places=3)

    def test_ch07_fit_matches_chapter(self):
        fit = F.ch07_fit()
        t = chapter("07")
        for size, n in (("2b", 701), ("4b", 687)):
            f = fit[size]
            self.assertEqual(len(f["xs"]), n)
            for v in (f"{f['slope_s_per_ktok']:.3f}", f"{f['intercept_s']:.2f}", f"{f['r']:.3f}",
                      f"{f['first_step_s']:.2f} 秒", f"{f['sum_step_s']:,.2f} 秒", f"{round(f['median_tok_per_s']):,}"):
                self.assertIn(v, t, (size, v))
            self.assertLess(abs(f["mem_r"]), 0.05)
            self.assertGreater(f["r"], 0.8)

    def test_svgs_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            F.render(tmp)
            for stem in STEMS:
                got = (Path(tmp) / f"{stem}.svg").read_bytes()
                self.assertEqual(got, (ROOT / f"assets/images/plots/{stem}.svg").read_bytes(), stem)

    def test_mutation_is_detected(self):
        # Perturbing one metrics row must change the fit printed in the chapter.
        orig = F.load
        def patched(name):
            d = orig(name)
            if name == "ch07_sft_metrics.json":
                d["runs"]["2b"][5]["step_time_s"] += 500
            return d
        F.load = patched
        try:
            slope = f"{F.ch07_fit()['2b']['slope_s_per_ktok']:.3f}"
        finally:
            F.load = orig
        self.assertNotIn(f"| {slope} |", chapter("07"))
        m = json.loads((ROOT / "data/ch06_lora_scale.json").read_text())["models"]["2b"]
        m["lora_trainable_parameters"] *= 2
        self.assertNotIn(f"{F.lora_budget(m)['static']:.2f}", chapter("06"))


if __name__ == "__main__":
    unittest.main()
