#!/usr/bin/env python3
"""Render the measured chapter-zero chart from a minimal local data snapshot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE = "result/round10/conclusion.md 最终结项总结 §2.1（提交 ca57456）"
FONT = 'Noto Sans CJK SC'
COLORS = ['#386bb3', '#087f83', '#bd6920']


def render(out):
    # Refuse silent fallback: a missing CJK font otherwise produces unreadable labels.
    font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    data = json.loads((ROOT / 'data/ch00_metrics.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': FONT, 'font.size': 12, 'axes.unicode_minus': False,
                         'text.color': '#18243b', 'axes.labelcolor': '#18243b',
                         'xtick.color': '#18243b', 'ytick.color': '#18243b'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 6.8), dpi=160,
                                     gridspec_kw={'width_ratios': [1.65, 1]})
    fig.patch.set_facecolor('#f7f8fc')
    fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.28, wspace=.23)
    fig.text(.035, .94, '项目实测：Base、SFT 与 RL 的评测结果', fontsize=21, weight='bold')
    fig.text(.035, .875,
             'Qwen3.5-2B · Terminal-Bench 2.0 · 每阶段 89 个任务 × 5 次尝试 = 445 次试验', fontsize=12)
    for ax in (left, right):
        ax.set_facecolor('white')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e0e5ee')
        ax.tick_params(axis='both', length=0, pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylim(2.5, -.5)
        ax.set_yticks(range(3), [s['name'] for s in data['stages']])
    keys = ['pass_at_1', 'pass_at_3', 'pass_at_5']
    for j, (key, color) in enumerate(zip(keys, COLORS)):
        ys = [i + (j-1)*.23 for i in range(3)]
        vals = [float(s[key]) for s in data['stages']]
        left.barh(ys, vals, height=.18, color=color, label=['pass@1', 'pass@3', 'pass@5'][j])
        for y, value, stage in zip(ys, vals, data['stages']):
            left.text(value+.08, y, stage[key]+'%', va='center', color=color, fontsize=11)
    left.set_xlim(0, 5.2)
    left.set_xticks(range(6))
    left.set_title('任务通过率（%）', loc='left', fontsize=15, weight='bold', pad=42)
    left.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3,
                frameon=False, borderaxespad=0, fontsize=11, columnspacing=1.5)
    vals = [float(s['valid_tool_call_rate']) for s in data['stages']]
    right.barh(range(3), vals, height=.26, color=COLORS[1])
    for i, (value, stage) in enumerate(zip(vals, data['stages'])):
        right.text(value+2, i, stage['valid_tool_call_rate']+'%', va='center', color=COLORS[1], fontsize=11)
    right.set_xlim(0, 122)
    right.set_xticks([0, 50, 100])
    right.set_title('有效工具调用率（%）', loc='left', fontsize=15, weight='bold', pad=42)
    fig.text(.035, .18, '本次 SFT 的三项通过率均高于 RL；工具调用格式有效不等于任务完成。', fontsize=12)
    fig.text(.035, .135, '两图使用不同横轴范围。仅展示本项目结果，不代表通用算法排名。', fontsize=12)
    fig.text(.035, .085, '来源：'+DATA_SOURCE, fontsize=11)
    fig.text(.035, .04, '百分数保留来源显示精度；全部指标来自本地脱敏快照 data/ch00_metrics.json。', fontsize=11)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'ch00_metrics.png'
    fig.savefig(target, dpi=160, metadata={'Software': 'matplotlib', 'Description': DATA_SOURCE})
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'assets/images/plots', help='输出目录')
    args = parser.parse_args()
    print(render(args.out))


if __name__ == '__main__':
    main()
