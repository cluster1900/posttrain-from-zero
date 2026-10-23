"""Render round5 sandbox probe elapsed times from committed JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_SOURCE = "固定实验提交 ca57456：result/round5/sandbox_result.json + summary.json"
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ch10_sandbox_probe.json"


def render(out: Path) -> None:
    obj = json.loads(DATA.read_text(encoding="utf-8"))
    cmds = obj["commands"]
    labels = [c["label"] for c in cmds]
    vals = [float(c["elapsed_s"]) for c in cmds]
    lo, hi = 0.0, max(vals + [1.0])
    span = max(hi - lo, 1e-12)
    bars = []
    n = len(vals)
    for i, (label, v) in enumerate(zip(labels, vals)):
        x = 80 + i * (620 / n)
        bh = 150 * (v - lo) / span
        y = 250 - bh
        bw = 620 / n * 0.6
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="#1565c0"/>'
            f'<text x="{x + bw/2:.1f}" y="272" font-size="10" text-anchor="middle">{label}</text>'
            f'<text x="{x + bw/2:.1f}" y="{y - 6:.1f}" font-size="11" text-anchor="middle">{v:.3f}s</text>'
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    create_s = obj["sandbox_create_s"]
    (out / "ch10_sandbox_probe.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="320" '
        f'font-family="Noto Sans CJK SC, DejaVu Sans, sans-serif">'
        f'<title>round5 sandbox probe elapsed_s</title>'
        f'<desc>{DATA_SOURCE}; sandbox_create_s={create_s}</desc>'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="24" y="24" font-size="16">round5 固定探针命令耗时（秒）</text>'
        f'<text x="24" y="48" font-size="12">sandbox_create_s={create_s}; timeout 映射见 summary</text>'
        f'<line x1="70" y1="250" x2="690" y2="250" stroke="#333"/>'
        f'{"".join(bars)}'
        f'<text x="70" y="300" font-size="11">来源：{DATA_SOURCE}</text>'
        f"</svg>\n",
        encoding="utf-8",
    )
    print(f"wrote ch10_sandbox_probe.svg under {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "assets/images/plots")
    render(parser.parse_args(argv).out)


if __name__ == "__main__":
    main()
