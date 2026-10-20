"""Reader experiment parameter counts, not project training metrics."""
DATA_SOURCE = 'examples/ch06_model_structure.py；两层微型模型 CPU 读者实验'
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
    data = json.loads((ROOT / 'data/ch06_tiny_parameters.json').read_text())
    values = [data[k] for k in ['base_unique_parameters', 'embedding_unique_parameters', 'lora_trainable_parameters']]
    fig, ax = plt.subplots(figsize=(11, 5.7), dpi=160)
    fig.subplots_adjust(left=.19, right=.93, top=.76, bottom=.23)
    bars = ax.barh(['唯一基座参数', '其中：embedding', 'LoRA 可训练增量'], values,
                   color=['#386bb3', '#8a9fb9', '#087f83'])
    ax.invert_yaxis()
    ax.set_xlim(0, 18500000)
    ax.set_xticks([0, 5000000, 10000000, 15000000], ['0', '500万', '1000万', '1500万'])
    ax.set_xlabel('唯一参数个数（线性坐标）')
    for bar, value in zip(bars, values):
        ax.text(value+150000, bar.get_y()+bar.get_height()/2, f'{value:,}', va='center', fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.06, .91, '【读者运行所得】微型模型参数量', fontsize=22, weight='bold')
    fig.text(.06, .83, '2 层、隐藏宽度 64；不是完整 2B。embedding 已包含于基座，不能再次相加。', fontsize=12)
    fig.text(.06, .10, 'LoRA 占包装后总参数约 0.8925%；共享输出头只计一次。', fontsize=12)
    fig.text(.06, .045, '来源：'+DATA_SOURCE, fontsize=11)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'ch06_parameters.png', metadata={'Software': 'matplotlib', 'Description': DATA_SOURCE})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT / 'assets/images/plots')
    render(parser.parse_args().out)
