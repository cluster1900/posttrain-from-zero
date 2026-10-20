# 批次 3 证据索引

范围：第 5–7 章。归档来源固定到 `ca57456`；正文只给路径与提交，不链接私有仓库。历史执行代码与归档提交分别记录。这里的 CPU 实验不是历史训练结果复现。

| 章节／事实 | 来源路径及版本 | 核验边界 |
| --- | --- | --- |
| 5：模板前缀、观测、解析、整段与增量合同 | `train/chat_format.py`、`tests/test_chat_format.py`、`configs/chat_templates/qwen35_agent.jinja`，`ca57456` | 教学例子只验证自包含文本合同；不替代真实 tokenizer 测试 |
| 5：foreign stop 修复 | 历史提交 `33f3cc5` 的 `train/rollout.py` 改动；`docs/apexo/06_risks_and_decisions.md`，`ca57456` | 历史生成停止符修复与后期 Harbor R48 分开叙述 |
| 6：目标层与冻结头 | `train/model_utils.py`、`tests/conftest.py`、`tests/test_model_utils.py`、`docs/apexo/05_testing_plan.md`，`ca57456` | 12 类名称不等于实例数；本书 tiny 模型有 15 个实例 |
| 6：参数量 | `examples/ch06_model_structure.py` 的本书 CPU 实验 | 实际构造随机两层模型并安装 PEFT，不是完整 2B/4B 数量；可选固定 revision 配置下载未实跑 |
| 7：配置和完整 batch 计数 | `configs/sft/qwen35_2b_lora.yaml`、`qwen35_4b_lora.yaml`、`train/sft_trainer.py`，`ca57456`；配置另核历史 `f5abb46` | 历史配置与归档配置的核心数值一致；包装脚本覆盖 save_every 和 export_merged |
| 7：真实入口／时刻 | `result/round2/README.md`、`result/round2/scripts/run_formal_sft.py`、`result/round2/logs/formal-sft.log`、`result/round2/results/formal_sft/colab_sft_2b.log`、`colab_sft_4b.log` 及两份 `sft_handoff_*.json`，`ca57456` | 正式执行代码 `f5abb46`；按训练循环日志计算间隔，不等于费用或全流程墙钟 |
| 7：smoke／正式长度 pilot／坑 | `docs/apexo/00_progress.md`、`docs/apexo/06_risks_and_decisions.md`、`result/round2/results/memory_audit_20261013.md`、`result/round2/results/sft_len_2b_metrics.jsonl`、`sft_len_4b_metrics.jsonl`，`ca57456` | 不同日期、阶段、单位与内存口径不拼接；缺失结束日志不补造 |
| 7：原始曲线 | `result/round2/results/formal_sft/metrics_2b.jsonl`、`metrics_4b.jsonl`，`ca57456` | 702／688 行八个数值字段逐行保留，无平滑；详见 data/README.md |

## 三段新增静态 Python 摘录

- 第 5 章：`train/chat_format.py` L190–194，`ca57456`。
- 第 6 章：`train/model_utils.py` L119–122，`ca57456`。
- 第 7 章：`train/sft_trainer.py` L190–194，`ca57456`。

三段均为连续原样摘录，无敏感标识替换，以 `.source-excerpt` 隔离，不能误作独立示例运行。第 7 章 Bash 块是明确省略私人目录的历史入口汇总，不声称为连续逐字摘录。公开技术出处与许可保留在正文和 `chapters/99_credits.md`；未复制第三方完整模板、系统提示词或数据轨迹。

原始数值文件 SHA-256（用于核对归档，不是新实验）：

- `metrics_2b.jsonl`：`484b57fe42f58c6d7a12e5372723cee58c0e861c999934e044df37fb41b4a1ff`。
- `metrics_4b.jsonl`：`e3947c0517e72080b21ac5a44a54c3de53275379723b7630d69c07bc02d43819`。
