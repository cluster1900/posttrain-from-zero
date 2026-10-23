"""Render chapter-9 data plots from committed RL metrics JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_SOURCE = (
    "固定实验提交 ca57456：result/round2/results/rl_2b_24k_metrics.jsonl；"
    "result/round10/rl-2b/continue-final/metrics.jsonl；"
    "result/round10/rehearsal-2b/metrics.jsonl"
)
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ch09_rl_metrics.json"


def chart(title, labels, values, out: Path, ylabel: str):
    vals = [float(v) for v in values]
    lo = min(vals + [0.0])
    hi = max(vals + [1.0])
    span = max(hi - lo, 1e-12)
    pts = []
    for i, v in enumerate(vals):
        x = 70 + 620 * i / max(1, len(vals) - 1)
        y = 250 - 150 * (v - lo) / span
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    marks = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1565c0"/>' for x, y in pts
    )
    labs = "".join(
        f'<text x="{70 + 620 * i / max(1, len(labels) - 1):.1f}" y="272" '
        f'font-size="11" text-anchor="middle">{label}</text>'
        for i, label in enumerate(labels)
    )
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" '
        f'font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">'
        f'<title>{title}</title><desc>{DATA_SOURCE}; {ylabel}</desc>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="24" y="24" font-size="16">{title}</text>'
        f'<line x1="70" y1="250" x2="690" y2="250" stroke="#333"/>'
        f'<line x1="70" y1="100" x2="70" y2="250" stroke="#333"/>'
        f'<polyline points="{poly}" fill="none" stroke="#1565c0" stroke-width="3"/>'
        f"{marks}{labs}"
        f'<text x="70" y="300" font-size="11">{ylabel}; min={lo:.6g} max={hi:.6g}</text>'
        f"</svg>\n",
        encoding="utf-8",
    )


def chart_multi(title, labels, series, out: Path, ylabel: str):
    all_values = [float(v) for values in series.values() for v in values]
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
        f'<text x="{70 + 620 * i / max(1, len(labels) - 1):.1f}" y="272" '
        f'font-size="11" text-anchor="middle">{label}</text>'
        for i, label in enumerate(labels)
    )
    legend = "".join(
        f'<text x="{70 + 170 * i}" y="300" font-size="11" fill="{colors[i]}">{name}</text>'
        for i, name in enumerate(series)
    )
    out.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330" '
        f'font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">'
        f'<title>{title}</title><desc>{DATA_SOURCE}; {ylabel}</desc>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="24" y="24" font-size="16">{title}</text>'
        f'<line x1="70" y1="250" x2="690" y2="250" stroke="#333"/>'
        f'<line x1="70" y1="100" x2="70" y2="250" stroke="#333"/>'
        f'{"".join(lines)}{labs}{legend}'
        f'<text x="70" y="320" font-size="11">{ylabel}; min={lo:.6g} max={hi:.6g}</text>'
        f"</svg>\n",
        encoding="utf-8",
    )


def render(out: Path) -> None:
    obj = json.loads(DATA.read_text(encoding="utf-8"))
    cont = obj["runs"]["round10_continue_final"]["records"]
    rehe = obj["runs"]["round10_rehearsal_2b"]["records"]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [r["step"] for r in cont]
    useful_ratio = [
        (r["useful_groups"] / r["groups"]) if r["groups"] else 0.0 for r in cont
    ]
    chart(
        "continue-final：reward_mean",
        labels,
        [r["reward_mean"] for r in cont],
        out / "ch09_reward_useful.svg",
        "reward_mean；useful比=" + ",".join(f"{x:.2f}" for x in useful_ratio),
    )
    chart(
        "continue-final：DPPO kept_frac",
        labels,
        [r["kept_frac"] for r in cont],
        out / "ch09_kept_frac.svg",
        "train/kept_frac",
    )
    chart(
        "continue-final：sync/logprob_abs_diff",
        labels,
        [r["sync_logprob_abs_diff"] for r in cont],
        out / "ch09_sync_diff.svg",
        "sync/logprob_abs_diff",
    )
    chart_multi(
        "continue-final：gen/train/step 时间",
        labels,
        {
            "gen_s": [r["gen_s"] for r in cont],
            "train_s": [r["train_s"] for r in cont],
            "step_s": [r["step_s"] for r in cont],
        },
        out / "ch09_time_split.svg",
        "seconds",
    )
    chart_multi(
        "rehearsal-2b：无沙箱短链路 step 时间",
        [r["step"] for r in rehe],
        {
            "gen_s": [r["gen_s"] for r in rehe],
            "train_s": [r["train_s"] for r in rehe],
            "step_s": [r["step_s"] for r in rehe],
        },
        out / "ch09_rehearsal_time.svg",
        "seconds",
    )
    print(f"wrote 5 data-derived SVG plots under {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets/images/plots")
    render(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
