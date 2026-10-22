#!/usr/bin/env python3
"""Render four chapter-9 plots from the sanitized real metrics JSON."""
import json
from pathlib import Path
DATA_SOURCE = "fixed experiment commit ca57456: result/round2/results/rl_2b_24k_metrics.jsonl"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/ch09_rl_2b_24k_metrics.json"

def chart(title, labels, values, out, ylabel):
    vals=[float(v) for v in values]
    lo=min(vals+[0.0]); hi=max(vals+[1.0]); span=max(hi-lo,1e-12)
    pts=[]
    for i,v in enumerate(vals):
        x=70+620*i/max(1,len(vals)-1); y=250-150*(v-lo)/span
        pts.append((x,y))
    poly=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    marks="".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1565c0"/>' for x,y in pts)
    labs="".join(f'<text x="{70+620*i/max(1,len(labels)-1):.1f}" y="272" font-size="11" text-anchor="middle">{str(l)}</text>' for i,l in enumerate(labels))
    out.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320"><title>{title}</title><desc>{DATA_SOURCE}; {ylabel}</desc><rect width="100%" height="100%" fill="white"/><text x="24" y="24" font-size="16">{title}</text><line x1="70" y1="250" x2="690" y2="250" stroke="#333"/><line x1="70" y1="100" x2="70" y2="250" stroke="#333"/><polyline points="{poly}" fill="none" stroke="#1565c0" stroke-width="3"/>{marks}{labs}<text x="70" y="300" font-size="11">{ylabel}; min={lo:.6g} max={hi:.6g}</text></svg>\n')


def chart_multi(title, labels, series, out, ylabel):
    all_values = [float(value) for values in series.values() for value in values]
    lo = min(all_values + [0.0])
    hi = max(all_values + [1.0])
    span = max(hi - lo, 1e-12)
    colors = ["#1565c0", "#ef6c00", "#2e7d32"]
    lines = []
    for index, (name, values) in enumerate(series.items()):
        points = []
        for i, value in enumerate(values):
            x = 70 + 620 * i / max(1, len(labels) - 1)
            y = 250 - 150 * (float(value) - lo) / span
            points.append((x, y))
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        marks = "".join(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[index]}"/>'
            for x, y in points
        )
        lines.append(
            f'<polyline data-series="{name}" points="{poly}" fill="none" '
            f'stroke="{colors[index]}" stroke-width="3"/>{marks}'
        )
    labs = "".join(
        f'<text x="{70+620*i/max(1,len(labels)-1):.1f}" y="272" font-size="11" '
        f'text-anchor="middle">{label}</text>' for i, label in enumerate(labels)
    )
    legend = "".join(
        f'<text x="{70+170*i}" y="300" font-size="11" fill="{colors[i]}">{name}</text>'
        for i, name in enumerate(series)
    )
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330">'
        f'<title>{title}</title><desc>{DATA_SOURCE}; {ylabel}</desc>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="24" y="24" font-size="16">{title}</text>'
        f'<line x1="70" y1="250" x2="690" y2="250" stroke="#333"/>'
        f'<line x1="70" y1="100" x2="70" y2="250" stroke="#333"/>'
        f'{"".join(lines)}{labs}{legend}'
        f'<text x="70" y="320" font-size="11">{ylabel}; min={lo:.6g} max={hi:.6g}</text></svg>\n',
        encoding="utf-8",
    )

def main():
    obj=json.loads(DATA.read_text()); rows=obj["records"]; labels=[r["step"] for r in rows]
    out=ROOT/"assets/images/plots"; out.mkdir(parents=True,exist_ok=True)
    chart("Reward and useful-group ratio",labels,[r["reward_mean"] for r in rows],out/"ch09_reward_useful.svg","reward_mean; useful_groups/groups=2/2 for each step")
    chart("DPPO kept fraction",labels,[r["kept_frac"] for r in rows],out/"ch09_kept_frac.svg","train/kept_frac")
    chart("Synchronization logprob drift",labels,[r["sync_logprob_abs_diff"] for r in rows],out/"ch09_sync_diff.svg","sync/logprob_abs_diff")
    chart_multi("Generation, training, and step time", labels, {"gen_s": [r["gen_s"] for r in rows], "train_s": [r["train_s"] for r in rows], "step_s": [r["step_s"] for r in rows]}, out/"ch09_time_split.svg", "seconds")
    print("wrote 4 data-derived SVG plots")

if __name__=="__main__":
    main()
