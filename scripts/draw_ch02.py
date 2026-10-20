"""Original chapter-two conceptual diagrams; no measured values."""
from draw_ch01 import SVG, flow


def main():
    s = SVG('终端智能体：行动、观测与最终验证', 400)
    boxes = [(40, '任务与上下文', ['目标、历史行动', '工具返回的观测']),
             (390, '模型 / 策略', ['生成 bash 工具调用', '不是直接改动文件']),
             (740, '隔离环境', ['执行行动，改变状态', '提交时触发 verifier'])]
    for x, title, lines in boxes:
        s.rect(x, 110, 290, 120);s.text(x+145, 140, title, 23, 'middle', weight='bold')
        for i,line in enumerate(lines):s.text(x+145,177+i*28,line,19,'middle')
    s.line('M333 167 H383',True);s.line('M683 167 H733',True)
    s.line('M885 233 V290 H535 V237',True)
    s.text(710,320,'观测返回模型，继续下一轮',20,'middle')
    s.text(40,365,'流程示意：最终奖励来自 verifier，不是模型宣称自己成功。',20)
    s.save('ch02_agent_loop.svg')
    flow('ch02_artifacts.svg','流程中的产物与证据',[
        ('固定数据',['manifest 与哈希','内容与版本可核验']),
        ('SFT 预热',['成功轨迹 → adapter','监督区段有明确规则']),
        ('真实环境 RL',['rollout 与新 adapter','奖励 / 同步 / 更新记录']),
        ('独立评测',['trial 与阶段报告','指标与判定口径分开'])],
        '职责示意，不代表每个阶段都提高性能；导出权重不是完成评测。')


if __name__ == '__main__':main()
