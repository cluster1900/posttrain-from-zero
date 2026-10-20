"""Original chapter5 text-contract diagrams; no measured values."""
from draw_ch01 import flow


def main():
    flow('ch05_contract.svg', '对话不是一段无结构字符串：四层分别核对', [
        ('消息对象', ['角色与顺序', '工具调用及返回']),
        ('模板文本', ['角色边界与换行', '思考、调用、停止符']),
        ('分词后的序列', ['固定 tokenizer 版本', '追加边界与 token ID']),
        ('训练目标', ['labels 与 loss mask', '只监督允许的区段']),
    ], '结构示意。文本相等不自动证明分别分词后的片段可直接拼接。')
    flow('ch05_incremental.svg', '增量路径与整段渲染：相同历史，相同边界', [
        ('初始消息', ['渲染生成前缀', '进入 assistant 思考段']),
        ('增量追加', ['真实生成文本', '观测分组与下一轮前缀']),
        ('独立整段渲染', ['完整消息列表', '末轮包含结束换行']),
        ('比较合同', ['full = incremental + 换行', '另测 token 及 mask']),
    ], '两条构造路径在最后比较；图中步骤描述验证顺序，不表示运行耗时。')


if __name__ == '__main__':
    main()
