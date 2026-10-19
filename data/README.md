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
