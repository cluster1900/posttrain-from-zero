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

## ch06_tiny_parameters.json

本书读者实验，不是原项目的 2B/4B 参数统计。`examples/ch06_model_structure.py` 在 torch 2.10.0、Transformers 5.6.0、PEFT 0.18.1 下创建随机两层文本模型，并实际安装 LoRA 后按唯一参数统计；不下载权重。测试重新构造模型并逐项比较 JSON。`scripts/figures/ch06_parameters.py` 画图。嵌入参数是基座总量的子集，不把两者相加。

## ch07_sft_metrics.json

来源：`result/round2/results/formal_sft/metrics_2b.jsonl` 与 `metrics_4b.jsonl`，归档提交 `ca57456`。保留全部 702 / 688 行的原始八个数值字段，无删点、平滑、插值或补值，不包含任务文本或运行地址。`scripts/figures/ch07_sft_curves.py` 画原始 loss/梯度曲线；`scripts/figures/ch07_memory.py` 从原始日志取 allocated 峰值，不能解释为物理显存或最低硬件需求。正式运行代码为 `f5abb46`，归档 pin 与实跑 pin 分开记录。

## 第 8 章构建时长

`ch08_builds.json` 从 `result/round8/costs-and-retention.json`（提交 `ca57456`）只摘录三次状态、观测墙钟秒数与总和，去掉全部云资源标识与链接。金额为 null，不能把时长当账单时长。`scripts/figures/ch08_builds.py` 输出1920×1088像素，保留失败、取消与成功三次结果。

## Coding batch4 真实指标

| 文件 | 含义 |
| --- | --- |
| `ch08_builds.json` | round8 构建墙钟（已有） |
| `ch09_rl_metrics.json` | round2 + round10 continue-final + rehearsal 指标 |
| `ch10_sandbox_probe.json` | round5 沙箱探针耗时与超时摘要 |

## 第 5–7 章真实数据补充

| 文件 | 来源与生成方式 |
| --- | --- |
| `ch05_token_boundary.json` | 读者实验：`examples/ch05_token_boundary_audit.py` 用源项目 `train/dataset.py`（提交 `ca57456`）在 SFT 成功轨迹 parquet（revision `9d6ab74`，SHA-256 `7507b499…e38a5`）上逐条构造训练样本，Qwen3.5-2B-Base tokenizer revision `b1485b2`。只存聚合计数和一个最小 token 例子，不含任务文本。 |
| `ch06_lora_scale.json` | 参数量为读者在 meta device 上按官方 config（固定 revision）实例化所得；显存峰值摘自 round2 `sft_len_{2b,4b}_metrics.jsonl` 与 `formal_sft/metrics_{2b,4b}.jsonl`（`ca57456`），字段为 allocated 高水位。 |
| `ch07_sft_metrics.json`（已有） | 7.8 节的耗时拟合由 `scripts/figures/ch05_07_real_data.py` 直接从此文件计算，第 1 步单列。 |

四张图 `ch05_turns`、`ch05_token_boundary`、`ch06_lora_scale`、`ch07_time_vs_tokens` 都由 `scripts/figures/ch05_07_real_data.py` 确定性生成 SVG。
