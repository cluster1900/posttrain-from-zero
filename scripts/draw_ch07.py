"""Original checkpoint/transfer diagram, not an execution timing plot."""
from draw_ch01 import flow


if __name__ == '__main__':
    flow('ch07_checkpoint.svg', '训练恢复与产物交接：两份不同的合同', [
        ('本地更新', ['模型与优化器状态', '步数、随机状态、数据序']),
        ('完整 checkpoint', ['写入后标记完成', '未完整目录不作为候选']),
        ('复制与读回', ['大文件是否真的落盘', '逐项 SHA-256 核验']),
        ('接收端交接', ['版本／配置／数据一致', 'adapter 完整 ≠ 质量通过']),
    ], '结构示意。完成标记不替代跨存储读回核验；没有保存的日志不靠推测补齐。')
