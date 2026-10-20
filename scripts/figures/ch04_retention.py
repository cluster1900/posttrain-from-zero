"""Plot only the three recorded retention counts, without interpolation."""
DATA_SOURCE = 'docs/apexo/03_data_plan.md §3.3（提交 ca57456）'
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
    data=json.loads((ROOT/'data/ch04_retention.json').read_text())
    pts=data['points'];total=data['total_trajectories']
    fig,ax=plt.subplots(figsize=(10,5.4),dpi=160)
    fig.patch.set_facecolor('#f7f8fc');fig.subplots_adjust(left=.12,right=.96,top=.78,bottom=.22)
    x=range(len(pts));values=[p['retained'] for p in pts]
    ax.bar(x,values,color=['#386bb3','#087f83','#386bb3'],width=.52)
    ax.axhline(total,color='#bd6920',linestyle='--',linewidth=1.5)
    ax.set_ylim(0,6400);ax.set_yticks([0,2000,4000,6000]);ax.set_ylabel('保留轨迹数',fontsize=13)
    ax.set_xticks(list(x),[f"{p['max_seq_len']:,}" for p in pts]);ax.set_xlabel('最大序列长度（token）',fontsize=13)
    for i,v in enumerate(values):ax.text(i,v-340,f'{v:,}',ha='center',color='white',fontsize=15,weight='bold')
    ax.text(1,total+130,'冻结成功轨迹总数 5,795',ha='center',color='#9a4f10',fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=12)
    fig.text(.06,.91,'源记录：SFT 长度阈值与保留轨迹数',fontsize=20,weight='bold',color='#18243b')
    fig.text(.06,.84,'离散报告点；不插值，不代表每轮训练最终样本数。',fontsize=12)
    fig.text(.06,.075,'来源：'+DATA_SOURCE,fontsize=11)
    fig.text(.06,.03,'24,576 阈值：5,795 − 5,618 = 177 条超长轨迹。',fontsize=11)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);target=out/'ch04_retention.png'
    fig.savefig(target,dpi=160,metadata={'Software':'matplotlib','Description':DATA_SOURCE});plt.close(fig)
    return target


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'assets/images/plots')
    print(render(p.parse_args().out))
