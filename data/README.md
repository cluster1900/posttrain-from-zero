# 图表数据来源

`ch00_metrics.json` 是最小化、脱敏的项目实测数据快照，仅包含生成第 0 章评测图所需的指标。

- 来源路径：`result/round10/conclusion.md`
- 来源章节：最终结项总结 §2.1
- 来源提交：`ca57456`
- 模型与评测集：Qwen3.5-2B，Terminal-Bench 2.0。
- 每阶段 89 个任务，每任务 5 次尝试，共 445 次试验。
- 百分数以字符串保留来源展示精度，不从已舍入的小数重新换算。没有推算或补充测量值。

生成命令：`python scripts/figures/ch00_metrics.py --out assets/images/plots`

需要 matplotlib 和 Noto Sans CJK SC 字体。图表生成不访问私有源仓库或外部服务；四张原理示意图由 `scripts/draw_ch01.py` 单独生成。

## ch04_retention.json

来源：源仓库 `docs/apexo/03_data_plan.md` §3.3，提交 `ca57456`。仅复制报告中的成功轨迹总数 5,795 和三个离散长度阈值/保留条数：16,384/5,330、24,576/5,618、32,768/5,711。没有任务文本、镜像地址或私人资源标识。这里是源文档统计的摘录，不声称本书重新运行全量数据统计。`scripts/figures/ch04_retention.py` 绘制柱图，不插值或平滑。
