# 批次 2 证据索引

范围：第 2–4 章。结项资料固定在源项目提交 `ca57456`；早期代码另列历史提交。这里只公开路径、提交和必要的脱敏摘录，不公开私有仓库链接、云资源标识或日志全文。文档中的配置是配置证据，不自动等于一次实跑；文档统计不是本书重新运行全量数据得到的结果。

| 章节／事实 | 证据位置 | 核验方式和边界 |
| --- | --- | --- |
| 2：SFT → RL → 评测及串行生成／更新 | `docs/apexo/01_architecture.md`、`09_custom_training_design.md`；`ca57456` | 对照组件职责和配置，不把结构示意作为训练结果 |
| 2：2B 与 4B 配置、默认禁止原始 Base 直接 RL | `configs/rl/qwen35_2b.yaml`、`configs/rl/qwen35_4b.yaml`；`ca57456` | 逐项检查默认值；不补写没有记录的模型选型动机 |
| 2：最终 SFT／RL 指标与事后门槛 | `result/round10/conclusion.md`、`docs/apexo/11_timeline.md`；`ca57456` | 沿用首批核对后的原始结果，门槛不写成预注册 |
| 3：v1 占位回答、无真实更新及模拟评测回退 | `train/rl_trainer.py`、`evaluation/run_eval.py`；`047c932` | L152–155 四行源码摘录逐字核对；保留历史上下文，不执行该源码 |
| 3：早期 v2 配置和修复实现 | `configs/sft/`、`configs/rl/`、`train/`、`requirements.txt`；`fbf621e` | 更新路径摘录为 `train/rl_trainer.py` L190–194；与结项配置分开，历史版本不是当前安装建议 |
| 3：审核问题清单、R1–R19、后续订正 | `docs/apexo/00_progress.md` §1.2–1.3、§6.1–6.4；`06_risks_and_decisions.md`；`ca57456` | 明确持续更新台账的时间边界，不将后续完成项倒推到首日 |
| 3：CPU 73 项与 GPU 验证待办、时间 | 源项目 PR #1；提交 `047c932` 和 `fbf621e` 的元数据 | 提交间隔／PR 生命周期不是人时、训练耗时或计费时长 |
| 4：round1 实跑代码版本、起止时间 | `result/round1/results/data_handoff.json`、`result/round1/logs/prepare_all.log`；`ca57456` | 文档归档提交与实跑提交 `ffd4713c8ba57057641d765afdc2b80a1dd1a44a` 分开；起止相差 1203 秒 |
| 4：任务总量、split、revision、SHA256、零交集 | round1 的 `manifest_info.json`、`data_handoff.json`、SFT 成功轨迹 info JSON（同一 `results/` 目录内以 `.info.json` 结尾的唯一文件；历史前缀不在正文重复） | 公开必要的内容哈希与数量，不公开存储位置和真实任务清单 |
| 4：split 与失败关闭 | `scripts/data/prepare_apexo_rl.py` L55–62、`prepare_all.sh`、`verify_manifest.py`；`ca57456` | split 函数逐字静态摘录；教学例子使用虚构 ID，不读取实际数据 |
| 4：清洗计数和长度分布 | `docs/apexo/03_data_plan.md` §3.3；round1 的 SFT 成功轨迹 info JSON；`ca57456` | 标记／轮次／样本单位分开；长度图为文档记录点，不插值或伪装全量重算 |
| 3–4：金额边界 | `docs/apexo/11_timeline.md`“作者补充”；`ca57456` | 全项目约 200 美元不能分摊成审核或数据准备费用；Colab 不在该口径内 |

## 教学代码与静态摘录

`examples/ch02_route.py`、`ch03_audit.py`、`ch04_data_freeze.py` 是本书独立编写的 CPU 教学例子。正文普通 Python 代码块与对应 `chapter_snippet` 逐字比对并执行；`.source-excerpt` 只用于历史／项目代码和未执行命令，不能作为本书测试成功的证据。原始来源和第三方许可见附录 E。

## 图与数据

第 2 章 2 张、第 3 章 4 张、第 4 章 3 张 SVG 均为本书原创结构示意。第 4 章长度保留图来自 `data/ch04_retention.json`，由 `scripts/figures/ch04_retention.py` 生成；三点对应正文源表，分母为 5795。复现支持 `--out`，测试核对数值及 PNG 字节。没有新增训练结果、预算或性能测量。
