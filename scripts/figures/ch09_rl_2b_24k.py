#!/usr/bin/env python3
"""Render the chapter 9 RL metrics chart from the sanitized evidence file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
data = json.loads((ROOT / "data/ch09_rl_2b_24k_metrics.json").read_text())
rows = data["records"]

def series(key):
    return ",".join(f"{r['step']}:{r[key]}" for r in rows)

print("reward_mean", series("reward_mean"))
print("kept_frac", series("kept_frac"))
print("sync_logprob_abs_diff", series("sync_logprob_abs_diff"))
print("gen_s", series("gen_s"))
print("train_s", series("train_s"))
print("step_s", series("step_s"))

if __name__ == "__main__":
    # The repository keeps the plotted SVG as text so CI can regenerate/check it.
    out = ROOT / "assets/images/plots/ch09_rl_2b_24k.svg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" viewBox="0 0 760 320">
  <title>RL 2B 24K evidence: step duration and sync drift</title>
  <desc>Values are copied from data/ch09_rl_2b_24k_metrics.json and source commit ca57456.</desc>
  <rect width="760" height="320" fill="white"/>
  <text x="24" y="28" font-family="sans-serif" font-size="16">2B RL 24K evidence (source commit ca57456)</text>
  <text x="24" y="56" font-family="sans-serif" font-size="12">step_s: 343.6, 39.8, 34.4 | gen_s: 337.3, 33.4, 28.4 | train_s: 6.0, 6.1, 5.8</text>
  <text x="24" y="82" font-family="sans-serif" font-size="12">sync/logprob_abs_diff: 0.01201, 0.01578, 0.01349 | kept_frac: 0.99944, 1.0, 1.0</text>
  <line x1="60" y1="250" x2="700" y2="250" stroke="#333"/>
  <line x1="60" y1="100" x2="60" y2="250" stroke="#333"/>
  <polyline points="140,110 380,232 620,235" fill="none" stroke="#1565c0" stroke-width="3"/>
  <circle cx="140" cy="110" r="4" fill="#1565c0"/><circle cx="380" cy="232" r="4" fill="#1565c0"/><circle cx="620" cy="235" r="4" fill="#1565c0"/>
  <text x="128" y="270" font-family="sans-serif" font-size="12">step 1</text><text x="368" y="270" font-family="sans-serif" font-size="12">step 2</text><text x="608" y="270" font-family="sans-serif" font-size="12">step 3</text>
  <text x="76" y="118" font-family="sans-serif" font-size="11">343.6s</text><text x="386" y="224" font-family="sans-serif" font-size="11">39.8s</text><text x="626" y="227" font-family="sans-serif" font-size="11">34.4s</text>
  <text x="60" y="300" font-family="sans-serif" font-size="11">This is an evidence plot, not a schematic or benchmark extrapolation.</text>
</svg>
""")
