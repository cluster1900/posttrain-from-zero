"""Original data-freezing schematics, not measured distributions."""
from draw_ch01 import SVG, flow


def main():
    flow('ch04_freeze.svg','数据冻结：失败关闭，成功才交接',[
        ('固定输入',['revision 与任务材料','不等于完整可用']),
        ('内容校验',['文件存在 / 正文一致','失败时移除旧产物']),
        ('稳定划分',['seed + task_id 哈希','训练与留出不相交']),
        ('校验并交接',['manifest + SHA-256','读回核验实际字节'])],
        '流程示意：目录存在不是解包完成；摘要一致不是数据语义正确。')
    s=SVG('任务、轨迹与监督轮：三个计量层次',380)
    for x,title,lines in [(40,'任务',['决定 train / held-out 归属','一份任务可以有多次尝试']),
                           (430,'轨迹',['一次尝试的完整交互','成功轨迹也可能含格式错误轮']),
                           (820,'assistant 轮',['决定哪些 token 计 loss','不训练某轮 ≠ 删除整条轨迹'])]:
        s.rect(x,110,340,155);s.text(x+170,145,title,25,'middle',weight='bold')
        for i,line in enumerate(lines):s.text(x+170,195+i*35,line,19,'middle')
    s.line('M383 185 H423',True);s.line('M773 185 H813',True)
    s.text(40,325,'层级示意，不表示真实样本比例；任务数、轨迹数、轮数不能相加。',21)
    s.save('ch04_units.svg')
    flow('ch04_handoff.svg','交接不是复制一个文件名',[
        ('内容合同',['必需字段与任务材料','训练 / 留出隔离']),
        ('冻结身份',['代码与数据 revision','seed 与划分大小']),
        ('字节身份',['三份 manifest 哈希','解包完成标记']),
        ('接收端核验',['刷新后重新读回','重建也要比对哈希'])],
        '交接示意：没有独立读回证据时，不把“已经写入”当成“可靠落盘”。')


if __name__=='__main__':main()
