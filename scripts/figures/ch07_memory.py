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
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    fig.subplots_adjust(left=.12, right=.95, top=.75, bottom=.22)
    values = [max(row['peak_mem_gb'] for row in data[size]) for size in ['2b', '4b']]
    ax.bar(['2B 正式 SFT', '4B 正式 SFT'], values, width=.5, color=['#386bb3', '#087f83'])
    ax.set_ylim(0, 26)
    ax.set_ylabel('PyTorch allocated 高水位（GiB）')
    for i, value in enumerate(values):
        ax.text(i, value+.6, f'{value:.2f}', ha='center', fontsize=18)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.06, .91, '正式 SFT 的内存统计口径', fontsize=22, weight='bold')
    fig.text(.06, .83, '不是整卡物理峰值，不是最小需求；不能据此宣布 40GB 全流程已验收。', fontsize=12)
    fig.text(.06, .10, '两段实际运行于 A100 80GB；逐步字段取最大，未重新运行 GPU。', fontsize=12)
    fig.text(.06, .04, '来源：'+DATA_SOURCE, fontsize=10)
    fig.savefig(out / 'ch07_memory.png', metadata={'Software': 'matplotlib', 'Description': DATA_SOURCE})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'assets/images/plots')
    render(parser.parse_args().out)
