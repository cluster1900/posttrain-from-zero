# 第四批证据索引（第 8–10 章）

归档固定提交为 `ca57456`。私有材料只按路径和提交引用，不提供私有链接，不复制账户、资源或连接标识。历史代码改动单独列版本，避免把结项文件倒写为早期已运行版本。

| 公开内容 | 依据路径 | 核验点 |
| --- | --- | --- |
| 未执行的直接 VM 方案 | `result/round3/README.md` | 明确未创建，不补造运行耗时 |
| GPU探针能力与Docker失败 | `result/round4/README.md` | 小矩阵/小模型通过，容器层注册失败，完整模型栈未跑 |
| 原生shell局部能力与清理 | `result/round5/README.md`、`result/round5/summary.json` | 文件状态、cwd/env复位、超时映射失败、模板补清理 |
| 单卡容量与平台代码适配 | `result/round6/README.md`、`result/round7/README.md`、`docs/apexo/00_progress.md` §6.32–6.33 | 物理/逻辑口径、可调缓存；本地修改不冒充云运行 |
| 三次镜像构建数据图 | `result/round8/costs-and-retention.json` | 255.836/1590.938/1579.385秒，合计3426.159；金额null |
| 镜像失败分类与配置恢复 | `result/round8/README.md`、风险R47 | 取消不是已证实OOM；历史配置原字节恢复 |
| 四次提交失败与最终探针 | `result/round9/README.md` | 项目字段、E2策略、全局CPU、SSD；两个空提交0分 |
| 配额口径修正 | `docs/apexo/00_progress.md` §6.44–6.46、`result/round10/scripts/preflight_fit.py` | Compute拒绝不等于Vertex Spot不可用，双读回UTC/SGT |
| DPPO与优势合同 | `train/losses.py`、`train/rl_trainer.py`、`configs/rl/qwen35_2b.yaml`、`configs/rl/qwen35_4b.yaml` | 截断留baseline、infra先除、方向mask、全局分母 |
| 补采与同步 | `train/rl_trainer.py`、`train/rollout.py`、`docs/apexo/08_execution_contract.md` | bounded refill、sleep/wake、prefix cache、概率差异 |
| 远程任务边界 | `train/remote_sandbox.py`、`train/sandbox.py`、`train/agent_env.py`、`scripts/agent_platform/cpu_startup.sh` | 可信宿主操作/容器内模型命令、奖励与退出码、状态包装器 |
| 凭据与共享存储限制 | `docs/apexo/06_risks_and_decisions.md` R40、R44 | 无沙箱云凭据，不把设计合同夸成全面安全证明 |
| 作者成本与路线回顾 | `docs/apexo/11_timeline.md` 作者10-19答复 | 约200美元GCP、不含Colab；不虚构分项金额 |

## 静态摘录

以下 `.source-excerpt` 均为连续原样片段，没有替换标识，因为所选范围不含私有身份或云资源值；它们不在教学代码执行检查中运行。

- 第8章：`result/round10/scripts/preflight_fit.py` L48–54，`ca57456`。
- 第9章：`train/losses.py` L56–64，`ca57456`。
- 第10章：`train/sandbox.py` L147–153，`ca57456`。

历史补采改动 `1037af0` 和顺序多工具调用改动 `a5a8046` 作为版本演进依据，不把其提交时间当训练开始时间。公开算法DPPO、参考数据与harness、模型与库的归属继续保留在 `chapters/99_credits.md`。

## 读者实验边界

三个 `examples/ch08_preflight.py`、`ch09_dppo.py`、`ch10_sandbox_contract.py` 是本书新写的离线缩小例子。没有云资源创建、命令执行、模型下载或真实RL运行。预检对缺失字段更严格；沙箱分类器的“验证未完成”标签不是源评分函数的替代；本章图示不暗示新增GPU测量。
