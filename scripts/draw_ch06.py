"""Original model schematics; shapes/counts come from the separate reader plot."""
from draw_ch01 import SVG


def layer_pattern():
    s = SVG('层类型的 3∶1 排列，不是参数量或耗时比例', 330)
    for i, label in enumerate(['线性注意力', '线性注意力', '线性注意力', '全注意力']):
        x = 45 + 295 * i
        s.rect(x, 105, 225, 95, '#e5f3ee' if i < 3 else '#e6edfa')
        s.text(x + 112, 151, label, 24, 'middle')
        if i < 3:
            s.line(f'M{x+230} 151 H{x+282}', True)
    s.text(45, 248, '源项目架构示意：重复的层类型组合；不在这里断言完整模型总层数。', 20)
    s.text(45, 286, '本章 CPU fixture：仅 1 层线性 + 1 层全注意力，不能套用真实规模统计。', 20)
    s.save('ch06_layer_pattern.svg')


def attention():
    s = SVG('两种注意力路径：投影职责不同，共同连接 MLP', 500)
    rows = [(115, '全注意力', ['q / k / v 投影', '注意力与门控', 'o 输出投影']),
            (280, 'Gated DeltaNet', ['合并 qkv；z、a、b', '卷积／状态更新／门控', 'out 输出投影'])]
    for y, name, labels in rows:
        s.text(45, y - 35, name, 23, weight='bold')
        for i, label in enumerate(labels):
            x = 45 + 300 * i
            s.rect(x, y, 265, 85, '#e5f3ee')
            s.text(x+132, y+42, label, 19, 'middle')
            if i < 2:
                s.line(f'M{x+268} {y+42} H{x+292}', True)
        s.rect(970, y, 185, 85)
        s.text(1062, y+30, '共同的 MLP', 20, 'middle')
        s.text(1062, y+60, 'gate / up / down', 16, 'middle')
        s.line(f'M910 {y+42} H963', True)
    s.text(45, 418, 'a 参与衰减量，b 经 sigmoid 形成更新强度；不是同一个门。', 21)
    s.text(45, 460, '职责示意，省略残差、归一化与详细张量运算；不作为完整架构方程。', 19)
    s.save('ch06_attention.svg')


def module_tree():
    s = SVG('微型文本模型：冻结组件与 12 类 LoRA 目标', 580)
    s.rect(40, 85, 1120, 65, '#e8edf4')
    s.text(600, 117, 'embedding（冻结）→ 两层骨干 → norm（冻结）→ 共享 lm_head（冻结）', 21, 'middle')
    groups = [('全注意力：4 类', ['q_proj   k_proj', 'v_proj   o_proj']),
              ('GDN：5 类', ['in_proj_qkv   in_proj_z', 'in_proj_a   in_proj_b', 'out_proj']),
              ('每层 MLP：3 类', ['gate_proj', 'up_proj   down_proj'])]
    for i, (title, lines) in enumerate(groups):
        x = 40 + 380 * i
        s.rect(x, 205, 350, 225, '#e5f3ee')
        s.text(x+175, 242, title, 22, 'middle', weight='bold')
        for j, line in enumerate(lines):
            s.text(x+175, 295+43*j, line, 21, 'middle')
    s.text(40, 483, '绿色表示安装 LoRA 的投影类别；绿色框中的基座权重仍冻结，只训练 A/B 增量。', 21)
    s.text(40, 530, '12 是叶名称类别数。两层 MLP 重复出现，本章微型模型实际命中 15 个实例。', 21)
    s.save('ch06_module_tree.svg')


if __name__ == '__main__':
    layer_pattern()
    attention()
    module_tree()
