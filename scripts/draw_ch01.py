#!/usr/bin/env python3
"""Generate four original Chinese SVG schematics using only stdlib."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'images'
FONT = 'Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei, PingFang SC, sans-serif'
INK = '#18243b'
TEAL = '#087f83'

class SVG:
    def __init__(self, title, height=360):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" viewBox="0 0 1200 {height}" role="img">',
                      f'<title>{escape(title)}</title>',
                      f'<rect width="1200" height="{height}" fill="#f7f8fc"/>',
                      '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#52657e"/></marker></defs>',
                      f'<g font-family="{FONT}" fill="{INK}">']
        self.text(40, 40, title, 27, weight='bold')

    def text(self, x, y, value, size=20, anchor='start', color=INK, weight='normal'):
        # dy places the visual center of CJK glyphs near y in all target renderers.
        self.parts.append(f'<text x="{x}" y="{y}" dy="0.35em" font-size="{size}" text-anchor="{anchor}" fill="{color}" font-weight="{weight}">{escape(str(value))}</text>')

    def rect(self, x, y, w, h, fill='white', stroke='#a0b2cd', radius=10):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def line(self, path, arrow=False, color='#52657e'):
        end = ' marker-end="url(#arrow)"' if arrow else ''
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"{end}/>')

    def save(self, name):
        (OUT / name).write_text('\n'.join(self.parts + ['</g></svg>']) + '\n', encoding='utf-8')


def flow(name, title, boxes, footer):
    s = SVG(title, 310)
    for i, (head, lines) in enumerate(boxes):
        x = 40 + 292*i
        s.rect(x, 95, 244, 125)
        s.text(x+122, 125, head, 22, 'middle', weight='bold')
        for j, text in enumerate(lines):
            s.text(x+122, 162+j*27, text, 17, 'middle')
        if i < 3:
            s.line(f'M{x+247} 157 H{x+283}', True)
    s.text(40, 270, footer, 18)
    s.save(name)


def lora():
    s = SVG('LoRA 并行路径：冻结基座与低秩增量相加', 390)
    s.rect(40, 160, 150, 70)
    s.rect(330, 85, 365, 85)
    s.rect(330, 235, 365, 85)
    s.rect(1010, 160, 150, 70)
    s.parts.append('<circle cx="880" cy="195" r="32" fill="white" stroke="#a0b2cd"/>')
    s.text(115, 195, '输入 x', 23, 'middle')
    s.text(512, 111, '冻结的基座权重', 22, 'middle', weight='bold')
    s.text(512, 144, 'W x', 22, 'middle')
    s.text(512, 260, '可训练的低秩增量', 22, 'middle', weight='bold')
    s.text(512, 293, '(α / r) B A x', 22, 'middle')
    s.text(880, 195, '+', 32, 'middle')
    s.text(1085, 195, '输出 y', 23, 'middle')
    for path in ['M190 195 H250 V127 H325', 'M250 195 V277 H325',
                 'M695 127 H770 V175 L845 187', 'M695 277 H770 V215 L845 203', 'M912 195 H1005']:
        s.line(path, True)
    s.text(40, 357, '原理示意，非项目测量。A 的形状：r × d_in；B 的形状：d_out × r。基座权重保持冻结。', 19)
    s.save('ch01_lora.svg')


def label_shift():
    s = SVG('标签移位与损失掩码：位置 t 预测位置 t + 1', 590)
    s.text(40, 83, '位置从 0 开始；原始 logits 序列长度为 4；标签为 [-100, -100, 2, 3]。', 21)
    xs = [310, 545, 780, 1015]
    s.text(40, 137, '原始位置', 20)
    for i, x in enumerate(xs):
        s.text(x, 137, str(i), 22, 'middle', weight='bold')
    s.text(40, 202, '模型分数', 20)
    s.text(40, 329, '原始标签', 20)
    for i, x in enumerate(xs):
        s.rect(x-86, 173, 172, 58, '#ffffff' if i<3 else '#e8edf4')
        s.text(x, 202, f'logits[{i}]', 21, 'middle')
        val = [-100, -100, 2, 3][i]
        s.rect(x-86, 300, 172, 58, '#e8edf4' if val == -100 else '#e5f3ee')
        s.text(x, 329, str(val), 23, 'middle')
        if i < 3:
            s.line(f'M{x+32} 235 L{xs[i+1]-32} 295', True)
    s.text(460, 245, '忽略目标', 17, 'middle')
    s.text(700, 245, '监督目标', 17, 'middle', color=TEAL)
    s.text(935, 245, '监督目标', 17, 'middle', color=TEAL)
    s.text(1015, 400, '末位分数不参与配对', 18, 'middle')
    s.text(310, 400, '首位标签移位后移除', 18, 'middle')
    s.rect(40, 436, 1120, 76, '#ffffff')
    s.text(62, 460, '实际配对：logits[0:3]  ↔  labels[1:4] = [-100, 2, 3]；只有 2 个目标计入损失。', 21)
    s.text(62, 492, '位置 1 的分数预测标签 2；位置 2 的分数预测标签 3。不要在模型内部移位后再次移位。', 19)
    s.text(40, 552, '原理示意，非项目测量。-100 只忽略对应目标的损失，不会删除输入上下文，也不是注意力掩码。', 19)
    s.save('ch01_label_shift.svg')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    flow('ch00_pipeline.svg', '项目后训练流程：从冻结数据到可核验评测',
         [('数据冻结', ['确定训练与评测边界', '记录版本与数据来源']),
          ('LoRA SFT', ['监督微调', '学习目标输出格式']),
          ('Apexo RL', ['强化学习', '基于真实沙箱反馈']),
          ('评测与对比', ['Harbor 评测框架', 'Terminal-Bench 2.0'])],
         '流程示意，非项目测量。对 Base、SFT、RL 分阶段评测；训练完成并不保证指标提升。')
    flow('ch01_training_step.svg', '一次优化器更新：按监督词元数加权累积',
         [('统计监督词元', ['N 为本次更新的', '有效监督词元总数']),
          ('逐个微步反传', ['每个微步：', '负对数似然之和 / N']),
          ('汇总并裁剪梯度', ['所有微步完成后', '只做一次梯度裁剪']),
          ('更新与清零', ['优化器更新一次', '然后清空梯度'])],
         '原理示意，非项目测量。微步之间不更新参数、不清空梯度；归一化分母覆盖本次更新。')
    lora()
    label_shift()
    print('Generated 4 SVG schematics')

if __name__ == '__main__':
    main()
