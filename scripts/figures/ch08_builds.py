"""Reproduce sanitized original build durations; no cost extrapolation."""
DATA_SOURCE = 'result/round8/costs-and-retention.json（提交 ca57456）'
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
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','axes.unicode_minus':False})
    data=json.loads((ROOT/'data/ch08_builds.json').read_text())
    values=[b['observed_wall_seconds'] for b in data['builds']]
    fig,ax=plt.subplots(figsize=(12,6.8),dpi=160)
    fig.subplots_adjust(left=.20,right=.95,top=.76,bottom=.23)
    ax.barh(['第一次：失败','第二次：主动取消','第三次：成功'],values,color=['#ad584e','#b08032','#087f83'],height=.55)
    ax.invert_yaxis(); ax.set_xlim(0,1850); ax.set_xlabel('观测墙钟时长（秒）',fontsize=14)
    ax.tick_params(labelsize=14)
    for i,v in enumerate(values):ax.text(v+24,i,f'{v:.3f}',va='center',fontsize=16)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.06,.91,'镜像交付经过三次构建，不只一次成功',fontsize=23,weight='bold')
    fig.text(.06,.83,'原始秒数未修改；总和 3426.159 秒，约 57.1 分钟。不是账单时长。',fontsize=14)
    fig.text(.06,.11,'第二次在可选扩展编译期间主动取消；没有证据把原因写成 OOM。',fontsize=13)
    fig.text(.06,.05,'来源：'+DATA_SOURCE,fontsize=11)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'ch08_builds.png',metadata={'Software':'matplotlib','Description':DATA_SOURCE})
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'assets/images/plots')
    render(p.parse_args().out)
