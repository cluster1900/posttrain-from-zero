"""Render every archived SFT row; no smoothing or altered raw measurements."""
DATA_SOURCE = 'result/round2/results/formal_sft/metrics_{2b,4b}.jsonl（提交 ca57456）'
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]


def render(out):
    font_manager.findfont(font_manager.FontProperties(family='Noto Sans CJK SC'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Noto Sans CJK SC', 'axes.unicode_minus': False})
    data = json.loads((ROOT / 'data/ch07_sft_metrics.json').read_text())['runs']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160, sharex='col')
    fig.subplots_adjust(left=.08, right=.96, top=.82, bottom=.14, hspace=.27, wspace=.20)
    for col, size in enumerate(['2b', '4b']):
        rows = data[size]
        steps = [r['step'] for r in rows]
        axes[0, col].plot(steps, [r['loss'] for r in rows], color='#386bb3', linewidth=.8)
        axes[0, col].set_title(f'{size.upper()}：{len(rows)} 个原始步', fontsize=15)
        axes[0, col].set_ylabel('有效 token 平均 loss')
        axes[0, col].set_ylim(0, .8)
        axes[1, col].plot(steps, [r['grad_norm'] for r in rows], color='#087f83', linewidth=.8)
        axes[1, col].axhline(1, color='#bd6920', linestyle='--', linewidth=1, label='裁剪阈值 1.0')
        axes[1, col].legend(loc='upper right', fontsize=10)
        axes[1, col].set_ylabel('裁剪调用返回的总范数')
        axes[1, col].set_ylim(0, max(1.1, max(r['grad_norm'] for r in rows)*1.05))
        axes[1, col].set_xlabel('优化器更新步')
        for row in range(2):
            axes[row, col].grid(alpha=.15)
            axes[row, col].spines[['top', 'right']].set_visible(False)
    fig.text(.06, .94, '正式 SFT：原始 loss 与 grad_norm，不平滑', fontsize=23, weight='bold')
    fig.text(.06, .885, '横轴为各自训练步，不是墙钟或相同样本；loss 下降不等于任务通过率提高。', fontsize=12)
    fig.text(.06, .04, '来源：'+DATA_SOURCE, fontsize=11)
    fig.savefig(out / 'ch07_sft_curves.png', metadata={'Software': 'matplotlib', 'Description': DATA_SOURCE})
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'assets/images/plots')
    render(parser.parse_args().out)
