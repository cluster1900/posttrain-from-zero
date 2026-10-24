"""Render the chapter 5-7 backfill plots (deterministic SVG) from committed data JSON.

Every coordinate is computed from data/*.json; nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

DATA_SOURCE = (
    "data/ch05_token_boundary.json（读者在源项目 ca57456 的 train/dataset.py 上对 5,795 条公开 SFT 轨迹实跑）；"
    "data/ch06_lora_scale.json（round2 sft_len_*/formal_sft metrics，提交 ca57456；官方 config 固定 revision）；"
    "data/ch07_sft_metrics.json（result/round2/results/formal_sft/metrics_{2b,4b}.jsonl，提交 ca57456）"
)
ROOT = Path(__file__).resolve().parents[2]
FONT = "Noto Sans CJK SC, DejaVu Sans, sans-serif"
GIB = 2 ** 30
BLUE, TEAL, ORANGE, GREY, RED = "#386bb3", "#087f83", "#ef6c00", "#9e9e9e", "#c62828"


def load(name):
    return json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, title):
        self.w, self.h, self.parts = w, h, []
        self.title = title

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, anchor="start", weight="normal", fill="#222"):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
                 f'font-weight="{weight}" fill="{fill}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke="#333", width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, label=None):
        t = f"<title>{esc(label)}</title>" if label else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}">{t}</rect>')

    def circle(self, x, y, r, fill, opacity=0.45):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" fill-opacity="{opacity}"/>')

    def write(self, path: Path):
        path.write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}">'
            f"<title>{esc(self.title)}</title><desc>{esc(DATA_SOURCE)}</desc>"
            f'<rect width="100%" height="100%" fill="white"/>' + "".join(self.parts) + "</svg>\n",
            encoding="utf-8",
        )


def axes(svg, x0, y0, x1, y1, ymax, yticks, ylabel, fmt="{:g}"):
    svg.line(x0, y0, x1, y0)
    svg.line(x0, y0, x0, y1)
    for t in yticks:
        y = y0 - (y0 - y1) * t / ymax
        svg.line(x0 - 4, y, x0, y)
        svg.line(x0, y, x1, y, stroke="#e0e0e0")
        svg.text(x0 - 8, y + 4, fmt.format(t), 11, "end")
    svg.text(x0 - 48, (y0 + y1) / 2, ylabel, 12, "middle")


def bar_hist(svg, keys, counts, x0, y0, x1, y1, color, ymax, xlabel_every):
    n = len(keys)
    bw = (x1 - x0) / n
    for i, (k, c) in enumerate(zip(keys, counts)):
        h = (y0 - y1) * c / ymax
        svg.rect(x0 + i * bw + 0.5, y0 - h, max(bw - 1, 1), h, color, f"{k}: {c}")
        if k % xlabel_every == 0:
            svg.text(x0 + i * bw + bw / 2, y0 + 16, k, 11, "middle")


def nice_ticks(vmax, n=5):
    raw = vmax / n
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    ticks = [round(step * i, 6) for i in range(0, n + 2) if step * i <= vmax * 1.0001]
    return ticks, step * (len(ticks) - 1) if ticks[-1] >= vmax else step * len(ticks)


def ch05_turns(out: Path):
    d = load("ch05_token_boundary.json")
    hist = {int(k): v for k, v in d["assistant_turns_hist"].items()}
    keys = list(range(min(hist), max(hist) + 1))
    counts = [hist.get(k, 0) for k in keys]
    ticks, ymax = nice_ticks(max(counts))
    total = sum(counts)
    turns = [k for k in keys for _ in range(hist.get(k, 0))]
    med = statistics.median(turns)
    s = Svg(900, 420, "图 5-3 每条成功轨迹的 assistant 轮数分布")
    s.text(30, 32, f"{total:,} 条成功轨迹的 assistant 轮数（中位数 {med:g}，最多 {max(keys)}）", 17, weight="bold")
    axes(s, 90, 340, 870, 70, ymax, ticks, "轨迹条数")
    bar_hist(s, keys, counts, 90, 340, 870, 70, BLUE, ymax, 5)
    s.text(480, 380, "每条轨迹中 assistant 消息条数（一轮 = 一次思考 + 一次 bash 调用）", 12, "middle")
    cap = hist[max(keys)]
    s.text(860, 340 - 270 * cap / ymax - 8, f"{max(keys)} 轮：{cap} 条", 11, "end", fill=RED)
    s.text(30, 408, "数据：data/ch05_token_boundary.json；源 train/dataset.py@ca57456，数据 revision 9d6ab74", 10, fill="#555")
    s.write(out / "ch05_turns.svg")


def ch05_token_boundary(out: Path):
    d = load("ch05_token_boundary.json")
    hist = {int(k): v for k, v in d["extra_tokens_hist"].items()}
    cap = 30
    keys = list(range(0, cap + 1))
    counts = [hist.get(k, 0) for k in keys[:-1]] + [sum(v for k, v in hist.items() if k >= cap)]
    ticks, ymax = nice_ticks(max(counts))
    s = Svg(900, 440, "图 5-4 分段编码比整段编码多出的 token 数")
    s.text(30, 32, "同一条轨迹：分段编码（项目 SFT/rollout 口径）比整段一次编码多几个 token", 17, weight="bold")
    s.text(30, 56, f"{d['piece_differs_from_whole']:,} / {d['trajectories']:,} 条不同；"
                   f"{d['crossed_boundaries']:,} / {d['span_boundaries']:,} 个片段边界被整段编码的一个 token 跨越，"
                   f"全部位于空思考 &lt;think&gt;\\n|\\n&lt;/think&gt;".replace("&lt;", "<").replace("&gt;", ">"), 12)
    axes(s, 90, 350, 870, 80, ymax, ticks, "轨迹条数")
    n = len(keys)
    bw = 780 / n
    for i, (k, c) in enumerate(zip(keys, counts)):
        h = 270 * c / ymax
        s.rect(90 + i * bw + 0.5, 350 - h, bw - 1, h, GREY if k == 0 else ORANGE, f"{k}: {c}")
        if k % 5 == 0:
            s.text(90 + i * bw + bw / 2, 366, f"≥{k}" if k == cap else k, 11, "middle")
    s.text(480, 392, "多出的 token 数 = 被跨越的边界数（每条轨迹逐一核对相等）", 12, "middle")
    s.text(30, 428, "数据：data/ch05_token_boundary.json；tokenizer revision b1485b2；源 train/dataset.py L7–10@ca57456", 10, fill="#555")
    s.write(out / "ch05_token_boundary.svg")


def lora_budget(m):
    """Static GiB terms implied by the config: bf16 base, fp32 head copy, fp32 LoRA + grad + AdamW(2)."""
    base = m["base_unique_parameters"] * 2 / GIB
    head = m["embedding_unique_parameters"] * 4 / GIB
    lora = m["lora_trainable_parameters"] * 4 * 4 / GIB
    return {"base_bf16": base, "head_fp32": head, "lora_fp32_state": lora,
            "static": base + head + lora, "peak": m["formal_peak_mem_gb_max"],
            "rest": m["formal_peak_mem_gb_max"] - (base + head + lora)}


def ch06_lora_scale(out: Path):
    d = load("ch06_lora_scale.json")["models"]
    s = Svg(900, 440, "图 6-5 2B/4B 正式 SFT allocated 高水位拆解")
    s.text(30, 32, "正式 SFT 的 allocated 高水位：可由参数量算出的静态部分 vs 其余（激活与临时张量）", 16, weight="bold")
    ymax = 25
    axes(s, 120, 360, 640, 70, ymax, [0, 5, 10, 15, 20, 25], "GiB")
    colors = [("base_bf16", BLUE, "基座权重 bf16"), ("head_fp32", TEAL, "输出头 fp32 副本"),
              ("lora_fp32_state", ORANGE, "LoRA fp32 参数+梯度+AdamW"), ("rest", GREY, "其余：激活、临时张量")]
    for i, size in enumerate(["2b", "4b"]):
        b = lora_budget(d[size])
        x = 190 + i * 230
        y = 360
        for key, color, _ in colors:
            h = 290 * b[key] / ymax
            s.rect(x, y - h, 120, h, color, f"{key}={b[key]:.2f}")
            s.text(x + 128, y - h / 2 + 4, f"{b[key]:.2f}", 11)
            y -= h
        s.text(x + 60, y - 8, f"峰值 {b['peak']:.2f}", 12, "middle", weight="bold")
        m = d[size]
        s.text(x + 60, 380, f"{size.upper()}：LoRA {m['lora_trainable_parameters'] / 1e6:.1f}M"
                            f"（{m['trainable_fraction'] * 100:.2f}%）", 12, "middle")
    for j, (_, color, label) in enumerate(colors):
        s.rect(680, 110 + j * 28, 16, 16, color)
        s.text(704, 123 + j * 28, label, 12)
    s.text(30, 428, "数据：data/ch06_lora_scale.json；峰值取 formal_sft metrics 最大 peak_mem_gb（ca57456）；参数量为官方 config 的 meta 统计", 10, fill="#555")
    s.write(out / "ch06_lora_scale.svg")


def ols(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return b, my - b * mx, statistics.correlation(xs, ys)


def ch07_fit():
    runs = load("ch07_sft_metrics.json")["runs"]
    res = {}
    for size, rows in runs.items():
        r = rows[1:]  # step 1 includes compile/cache warm-up; reported separately
        xs = [q["label_tokens"] for q in r]
        ys = [q["step_time_s"] for q in r]
        b, a, rho = ols(xs, ys)
        res[size] = {"slope_s_per_ktok": b * 1000, "intercept_s": a, "r": rho,
                     "mem_r": statistics.correlation(xs, [q["peak_mem_gb"] for q in r]),
                     "first_step_s": rows[0]["step_time_s"], "xs": xs, "ys": ys,
                     "median_tok_per_s": statistics.median(x / y for x, y in zip(xs, ys)),
                     "sum_step_s": sum(q["step_time_s"] for q in rows)}
    return res


def ch07_time_vs_tokens(out: Path):
    fit = ch07_fit()
    s = Svg(900, 460, "图 7-4 正式 SFT 每步耗时与标注 token 数")
    s.text(30, 32, "正式 SFT：每步耗时随该步标注 token 数近似线性增长（第 1 步预热另计）", 16, weight="bold")
    x0, y0, x1, y1 = 90, 370, 870, 70
    xmax, ymax = 110000, 120
    axes(s, x0, y0, x1, y1, ymax, [0, 20, 40, 60, 80, 100, 120], "秒/步")
    for t in range(0, xmax + 1, 20000):
        x = x0 + (x1 - x0) * t / xmax
        s.line(x, y0, x, y0 + 4)
        s.text(x, y0 + 18, f"{t // 1000}k", 11, "middle")
    s.text((x0 + x1) / 2, y0 + 38, "该步全局标注 token 数（16 条样本之和）", 12, "middle")
    for size, color, ly in (("2b", BLUE, 92), ("4b", TEAL, 114)):
        f = fit[size]
        for x, y in zip(f["xs"], f["ys"]):
            s.circle(x0 + (x1 - x0) * x / xmax, y0 - (y0 - y1) * min(y, ymax) / ymax, 2.4, color)
        lo, hi = min(f["xs"]), max(f["xs"])
        p = [(x0 + (x1 - x0) * x / xmax, y0 - (y0 - y1) * (f["intercept_s"] + f["slope_s_per_ktok"] * x / 1000) / ymax) for x in (lo, hi)]
        s.line(p[0][0], p[0][1], p[1][0], p[1][1], stroke=color, width=2.5)
        s.rect(110, ly - 10, 12, 12, color)
        s.text(128, ly, f"{size.upper()}：{f['slope_s_per_ktok']:.3f} 秒/千 token，截距 {f['intercept_s']:.2f} 秒，r={f['r']:.3f}，"
                        f"n={len(f['xs'])}；第 1 步 {f['first_step_s']:.2f} 秒", 12)
    s.text(30, 450, "数据：data/ch07_sft_metrics.json ← result/round2/results/formal_sft/metrics_{2b,4b}.jsonl@ca57456；最小二乘直线，未平滑", 10, fill="#555")
    s.write(out / "ch07_time_vs_tokens.svg")


def render(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ch05_turns(out)
    ch05_token_boundary(out)
    ch06_lora_scale(out)
    ch07_time_vs_tokens(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "assets/images/plots")
    render(ap.parse_args().out)
