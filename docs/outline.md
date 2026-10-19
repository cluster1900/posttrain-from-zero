# 《后训练从 0 到 1》详细大纲（v2，待写作）

作者：Hawk Wu

本书根据作者的真实项目 `qwen-posttrain`（以下简称"源仓库"，当前为私有仓库）整理：在官方 `Qwen/Qwen3.5-2B-Base` / `Qwen3.5-4B-Base` 上，用作者的 **Apexo 后训练流程**（作者在阅读多方公开论文与开源实现的基础上自行设计和修改）做终端智能体（Terminal Agent）后训练，从 2026-10-10 第一次提交到 2026-10-19 结项。全书按"先打基础，再按时间讲真实过程"组织：每一步写清楚**做了什么、为什么这么做、遇到了什么坑、怎么解决、花了多久、之后怎么调整**。

## 写作规则

1. **每个事实都要能追溯**到源仓库的文件、日志、指标或提交（写明路径或 commit）。数字（loss、显存、耗时、pass@k）照抄原始记录，不四舍五入成别的意思。
2. **记录里没有的不编**。原因、心路、费用等如果找不到，就在章末"待作者确认"里列问题，不要替作者补动机。
3. **如实写结果**，包括不好看的数字：RL 后 pass@1（0.90%）比 SFT（1.12%）低，必须写。门禁按 Apexo 阈值描述（见第 10 章），所有实测数字照抄原始报告，一个都不改。不要写成"阈值在训练前就定好了"，记录里也没有这个说法。
4. **方法命名**：全书统一叫"Apexo 后训练流程""Apexo RL"等。不要把方法归属给某篇外部论文或某个外部项目。公开数据集和评测 harness 是固定版本的外部输入，在附录 E 按许可证署名。
5. 公开书稿里**不出现**云项目编号、bucket 名、Drive 文件夹 ID、服务器主机名、邮箱、密钥或任何凭据；用"项目 bucket""训练服务器"等代称。
6. 风格：中文、面向能写 Python 但没做过后训练的读者；概念先用直觉解释，再给最小代码；源仓库代码只摘关键片段并注明出处。
7. 时间：原始日志多为 UTC，正文统一写成"日期 + 时长"，需要具体时刻时注明时区。
8. **引用源仓库**：源仓库保持私有。书里以纯文本写出路径和提交号（例如 `docs/apexo/11_timeline.md`、提交 `ca57456`），不放读者打不开的链接，也不依赖读者去看源仓库。

## 章节写法（必须照做）

这本书要**写得很细**。读者只看书、不看源仓库，也要能明白每一步做了什么、为什么、花了多久、出了什么问题。

**训练阶段章节**（第 3、4、6、7、8、9、10、11 章）每章都必须有下面这些二级小节，标题照抄，顺序不限：

| 小节 | 要写什么 |
| --- | --- |
| `## 动机` | 这一步要解决什么问题，为什么在这个时间点做。只写记录里有的，或作者已确认的原因 |
| `## 配置` | 精确配置：模型和 revision、数据 revision 与条数、超参数（lr、batch、累积、LoRA r/alpha、max_seq_len、KV cache、vLLM 参数等）、机型与显存。照抄 YAML 或台账，注明文件路径 |
| `## 命令` | 实际跑过的命令（去掉项目编号、bucket、账户等，用代称），以及读者在 CPU 上能复现的最小命令 |
| `## 时间线与耗时` | 表格列出每次尝试的开始、结束（注明时区）和时长；时长由原始时间戳相减得到 |
| `## 坑与修复` | 每个坑一行，表头固定为 `\| 现象 \| 根因 \| 修复 \| 证据 \|`。证据列写源仓库路径、日志行或提交号；本阶段在 R1–R49 和 §6.x 里出现过的坑一个都不能漏 |
| `## 花费` | 用了什么资源（机型、GPU、Spot/普通、时长）。美元金额只有全项目总数（约 200 美元，全部是 Google Cloud），不要拆到各阶段，也不要估算 |
| `## 教训` | 这一步之后怎么调整的、对读者有什么可复用的经验；必须对应到上面写过的事实 |
| `## 本章出处` | 本章用到的全部源文件路径和提交号清单 |

其他章节（第 1、2、5、12、13 章和附录）至少要有 `## 本章出处`。前言除外。

**代码示例**：
- 放在 `examples/chNN_<主题>.py`，必须在 CPU 上几秒内跑完，不需要 GPU、网络或凭据。
- 每个示例都要有对应测试 `tests/test_chNN_<主题>.py`（pytest），并断言关键数值或形状。
- 正文里引用的示例代码必须和 `examples/` 里的文件一致。
- 依赖固定在 `requirements-examples.txt`。

## 验收标准（审核时逐条检查，任何一条不满足就退回）

| # | 标准 | 怎么检查 |
| --- | --- | --- |
| 1 | **每个数字都能追溯**：每个数字、时间、配置值都能在源仓库的文件或提交里找到原文；正文或「本章出处」写明路径或提交号 | 审核时每章至少抽查 10 个数字（不足 10 个则全查），只要有一个找不到原文就退回 |
| 2 | **不编造**：动机、费用、心路只写记录里有的或作者已确认的；门禁按 Apexo 阈值写，但不写成训练前预设；实测数字照抄 | 人工审核 |
| 3 | **训练阶段章节结构完整**：上面 8 个小节齐全，「坑与修复」用规定表头，每个坑都有证据 | `python3 scripts/check_book.py`（A3、A4）+ 人工核对 R1–R49 覆盖 |
| 4 | **代码示例能在 CPU 上跑且有测试** | CI 运行 `pytest`；`check_book.py` A5 检查每个示例都有测试 |
| 5 | **构建、Pages、EPUB 都通过** | 本地 `python3 build.py` 成功；PR 的 CI `build` job 绿（含 `check_book.py`、示例测试、`epubcheck` 校验 EPUB）；合并后 Pages 部署成功 |
| 6 | **没有旧方法名**：全书只用"Apexo"命名；外部数据集和 harness 的 ID 只出现在致谢与许可附录（文件名含 `credits`、`appendix_e` 或 `99_`） | `check_book.py` A1 |
| 7 | **没有秘密和项目标识**：没有云项目编号、bucket、服务账户、镜像仓库地址、邮箱、Drive 文件夹 ID、主机名、私钥或 API key | `check_book.py` A2 + 人工审核 |
| 8 | **写得够细**：读者不看源仓库也能复述这一步的做法、原因、耗时和每个坑 | 人工审核；明显只列要点、不讲过程的章节退回 |

PR 描述里要列出：本批每个数字/结论的出处、`check_book.py` 和 `build.py` 的输出、示例测试结果。

## 章节与素材映射

路径都相对于源仓库根目录，以源仓库 PR #6（提交 `ca57456`，Apexo 命名后的版本）为准。"§x" 指 `docs/apexo/00_progress.md` 的小节；"R" 编号指 `docs/apexo/06_risks_and_decisions.md` 的风险登记；时间线和作者补充见 `docs/apexo/11_timeline.md`。

### 第 0 章 前言：这本书讲的是哪个项目

- 项目目标、最终结果一览（Base / SFT / RL 三次全量评测）、全书路线图、怎么读。
- 素材：`README.md`、`AGENTS.md`、`result/round10/conclusion.md`（下半部分"最终结项总结"）。

### 第 1 章 PyTorch 入门：读懂后训练代码需要的那部分

不求全，只讲本项目代码里真正用到的：
1. 张量、dtype（fp32 / bf16）、device；显存是怎么被占用的（权重、梯度、优化器状态、激活）。
2. autograd 与一次训练步：前向 → loss → backward → optimizer.step；梯度累积（全局 batch 16 = bs 1 × 累积）。
3. `nn.Module`、`state_dict`、checkpoint 保存与恢复。
4. 省显存三件套：bf16、gradient checkpointing、LoRA（r=64、alpha=64，挂在哪些模块上）。
5. 交叉熵与"只对 assistant 段算 loss"的 mask；按 token 归一化。
6. 从 `transformers` 加载模型和分词器、chat template 是什么。
- 配套：每节一个能在 CPU 上跑的小例子；末尾对照源仓库 `train/model_utils.py`、`train/losses.py`、`train/sft_trainer.py`、`configs/sft/qwen35_2b_lora.yaml`。
- 素材：上述文件；`requirements.txt`（torch 2.10 / transformers 5.6.0 / peft 0.18.1 / vLLM 0.19.1）。

### 第 2 章 后训练全景与本项目的路线

- 预训练与后训练的区别；SFT、偏好优化（DPO）、在线 RL（PPO / GRPO / DPPO）各解决什么问题。
- 终端智能体任务：模型通过 bash 工具调用完成任务，环境里的 verifier 给 0/1 奖励。
- 本项目路线（Apexo 后训练流程）：公开数据冻结 → LoRA SFT 预热 → outcome-only DPPO（Apexo RL）→ Harbor + Terminal-Bench 2.0 评测；作者的设计取舍（单卡、LoRA、lr 等，相对常见多机方案）。
- 素材：`AGENTS.md`、`docs/apexo/01_architecture.md`、`docs/apexo/09_custom_training_design.md`（§1.3 设计取舍表）、`configs/rl/qwen35_2b.yaml` 注释、`docs/apexo/README.md` 流程图。
- 方法来源（作者 2026-10-19 确认）：Apexo 是作者在阅读多方公开论文与开源实现后自行设计和修改的流程。
- 模型规模（作者 2026-10-19 确认）：选 Qwen3.5 2B 是因为它**足够小**。4B 也试过：完成了 SFT，之后因成本太高停止（见第 12 章）。作者没有说明为什么选 Base 而不选 Instruct，书里只写"用的是官方 Base 模型"，不要替作者补理由。

### 第 3 章 第一版为什么不能用：从 v1 到 v2（10-10）

- v1 审核发现的 P0/P1 问题逐条讲：RL 训练器是空壳（硬编码 dummy response）、依赖还是 Qwen2.5、工具调用格式不是 Qwen3.5 原生、SFT 没有 collator、评测无结果时写假的 PASSED……每条讲"坑在哪 → 怎么发现 → v2 怎么改"。
- 教训：先审核再训练；"能跑"不等于"在训练"。
- 素材：§1.2 表、§1.3、§6.1–§6.4；提交 `047c932`、`fbf621e`；PR #1。

### 第 4 章 数据：把 1.5 万个任务冻结下来（10-12，round1）

- 数据源：公开的 15K 终端 RL 任务包与公开 SFT 轨迹集（具体数据集名、revision 和许可证见附录 E）；固定 revision、seed 42、留出 200 条；manifest + SHA-256。
- 数量与分布：rl_train 14,401 / heldout 200 / SFT 5,795 条；14,490 个不同镜像；SFT token 长度分位数与 `max_seq_len=24576` 的取舍；数据瑕疵（缺思考前缀、无工具调用轮、残留 `</think>`、非 bash 工具名）怎么处理。
- 过程：Colab CPU 上约 20 分钟跑完；D1/D2 校验。
- 素材：`docs/apexo/03_data_plan.md`、§2、`result/round1/`、`scripts/data/`、`tests/test_dataset.py`、`tests/test_manifest_and_eval.py`。

### 第 5 章 聊天模板与工具调用格式

- 为什么模板必须唯一、增量拼接要和整段渲染一致；Qwen3.5 原生 XML 工具调用；`<think>` 段。
- 坑：rollout 中训练了外来的 `<|endoftext|>` / eos 终止符（提交 `33f3cc5`）；官方 Base 只配了 `<|endoftext|>` 作 eos，vLLM 不停在 `<|im_end|>`，导致输出失控（R48，第 10 章展开）。
- 素材：`configs/chat_templates/qwen35_agent.jinja`、`train/chat_format.py`、`tests/test_chat_format.py`、R48。

### 第 6 章 SFT：在 Colab 上把格式教会（10-11 ~ 10-13，round2）

1. 小步验证：5 步 smoke（2B/4B loss 下降数字），再做正式长度短 SFT 和续训测试。
2. 正式 SFT：2 个 epoch；2B 702 步约 7 小时 45 分，4B 688 步约 11 小时 45 分；loss 曲线与显存（2B 13.91 GiB、4B 21.64 GiB，torch allocated）。
3. 坑与解决：
   - Drive 上的权重是 79 字节的符号链接替身，`safetensors` 报 `header too large` → 本地重建链接；
   - Drive 挂载约 67 分钟后掉线，进程卡死 → 材料先落 VM 本地盘；
   - causal-conv1d 与 CUDA 版本不匹配 → 跳过，走 torch 实现；
   - 40GB 上预演被"38 GiB 逻辑显存门禁"拦下但其实没 OOM → 改成物理显存采样门禁。
- 素材：`result/round2/README.md`、`result/round2/logs/formal-sft.log`、`result/round2/results/formal_sft/metrics_*.jsonl`、`result/round2/results/memory_audit_20261013.md`、§6.6–§6.8、§6.21–§6.27、`train/sft_trainer.py`、`train/dataset.py`。
- 建议配图：两条 SFT loss 曲线（由 metrics_*.jsonl 画，脚本放本书 `scripts/`）。

### 第 7 章 算力去哪里找：Colab、GCP 与 Agent Platform（10-11 ~ 10-14）

- 方案几次变化及原因：去掉 Daytona、拆 Colab/GCP（PR #4）→ 直接 GCP VM（round3，计划了但没执行）→ Agent Platform CustomJob + 临时 CPU 沙箱 VM（round4–round8）。
- 40GB 还是 80GB：§6.22、§6.32–§6.33 的决策过程；最后定为单卡 A100 40GB、训练与推理串行、KV cache 可调。
- 配额：A100 80GB 训练配额为 0；普通 A100 配额申请被拒；最后用 Vertex Spot A100 配额（R 记录与 §6.45–§6.46）。
- round9 的四次失败申请：项目编号字段用错、e2 机型不支持某维护策略、vCPU 超全局配额、pd-balanced 计入 SSD 配额——每次怎么分类、怎么修预检；"提交前两次只读核对"规则的由来。
- 镜像构建：causal-conv1d 源码编译卡住被取消（R47）。
- 素材：`result/round3/`–`result/round9/` 各 README、`docs/apexo/07_gcp_colab_a100_training_plan.md`、`docs/apexo/10_agent_platform_plan.md`、`result/round10/scripts/preflight_fit.py`、`result/round8/costs-and-retention.json`。
- 花费（作者 2026-10-19 确认）：全项目总花费约 **200 美元，全部是 Google Cloud**。仓库没有账单明细，书里只写这个总数，不要拆分到各轮。
- 为什么去掉 Daytona（作者 2026-10-19 确认）：Google Cloud 上已经有能承载沙箱的 VM，不需要再用 Daytona。

### 第 8 章 RL 原理与实现：DPPO 怎么在一张卡上跑

- 一步 RL 的完整流程：抽 4 个任务 × 每个 8 个样本 → vLLM 多轮 rollout（训练器离卡）→ 沙箱里执行 → verifier 0/1 奖励 → centered advantage → DPPO（binary TV 0.1）更新 LoRA → 权重推回 vLLM。
- 关键设计：零方差组过滤与有界 active sampling、截断样本 mask、`sync/logprob_abs_diff` 同步检查、k3 KL 只监控不惩罚（beta=0）、fp32 lm_head、分块 logprob 防 248K 词表 OOM、vLLM sleep/wake。
- 素材：`train/rl_trainer.py`、`train/rollout.py`、`train/losses.py`、`train/agent_env.py`、`configs/rl/qwen35_2b.yaml`、`docs/apexo/09_custom_training_design.md`、`docs/apexo/08_execution_contract.md`、`tests/test_rl_step.py`、`tests/test_losses.py`、提交 `1037af0`、`a5a8046`。

### 第 9 章 沙箱：让模型的命令只在该在的地方跑

- 为什么模型命令绝不能在宿主机裸跑；GPU worker 不跑 Docker；临时 CPU VM 跑原任务容器，经 IAP SSH 调用；VM 无服务账户、容器拿不到凭据（R44）。
- round5 原生 Shell Sandbox 实测（能力和限制），round9 原任务 verifier 探针（只证明链路通，不证明正确解法得 1 分）。
- 素材：`train/sandbox.py`、`train/remote_sandbox.py`、`scripts/agent_platform/cpu_startup.sh`、`result/round5/`、`result/round9/`、R40、R44。

### 第 10 章 评测：Harbor 与 Terminal-Bench 2.0 的长跑（10-14 ~ 10-17）

- 指标：89 个任务 × 5 次 = 445 trials；pass@1/3/5、valid_tool_call_rate。
- Apexo 门禁（`evaluation/run_eval.py` 默认值）：
  - SFT：有效工具调用率 ≥ 96%；pass@1 相对 Base 增益 ≥ +0.01，且 bootstrap 95% CI 下界 > 0。
  - RL：有效工具调用率 ≥ 96%；pass@1 相对 SFT 不劣于 −0.01（不要求提升）。
  - 训练健康检查：KL 最大值 ≤ 0.2、熵 ≤ 起始的 3 倍、无工具调用轮次 ≤ 10%。
  - 阈值修订于 10-19（素材：`result/round10/regate/README.md`）；照实写这个日期，不要写成训练前预设。
- Base 评测前三次失败：24576 上下文装不下 16384 输出 → 65536 被 4 小时超时掐断 → 18 小时那次 SSH 断线退出 255。
- 根因与修复：R48（eos 停止符 + `repetition_penalty=1.05` + `max_tokens=8192` + `command_timeout=300`）、R49（Harbor 后台脱机运行、隧道重连、磁盘清理、trial 级快照与跨作业恢复）。
- Spot 抢占：SFT 评测 289 → 377 → 445 三次才跑完。
- 耗时：Base v6 约 13 小时 43 分；SFT 评测三次的时长。
- 素材：`result/round10/README.md`、`result/round10/conclusion.md`（上半部分）、`result/round10/harbor-base/`、`result/round10/harbor-sft-2b/`、`scripts/agent_platform/harbor_eval.py`、`evaluation/run_eval.py`、R48、R49、§6.49–§6.50。

### 第 11 章 RL 实跑：八步，以及前两步为什么全是 0 分（10-17 ~ 10-18）

- Step 1 后被 4 小时超时取消 → 改 18 小时，从已提交 checkpoint 续训；Step 1–2 奖励全 0、梯度跳过；继续到 Step 8，5 步有非零奖励更新。
- 逐步指标表（reward_mean、useful_groups、loss、KL、entropy、显存、步时），步时 3,500–8,700 秒。
- 素材：`result/round10/rl-2b/`（`continue-final/metrics.jsonl`、`rollouts.jsonl`、各 run log）、`result/round10/sub2-orchestrator-status.json`、`result/round10/scripts/sub2_orchestrate.py`。

### 第 12 章 结果与诚实复盘

- 三次全量评测对比：pass@1 0.00% → 1.12% → 0.90%；valid_tool_call_rate 88.96% → 96.58% → 96.24%；各自解出了哪些任务。
- 门禁结论（Apexo 阈值）：SFT **PASSED**（有效调用率 96.58%；pass@1 增益 +0.0112，CI [+0.0022, +0.0247]）；RL **PASSED**（有效调用率 96.24%；pass@1 变化 −0.0022，CI [−0.0112, +0.0067]；kl_max 0.1979；无工具调用 8.23%）。素材：`result/round10/harbor-sft-2b/sft-gate-apexo.json`、`result/round10/harbor-rl-2b/rl-gate-apexo.json`。
- 讲清楚门禁通过能说明什么、不能说明什么：RL 只是"没有明显变差"，不是"变好了"。
- 只做了 8 步 RL、2B 单卡，结论边界；本项目不和任何外部报告的数字做比较。
- 4B 为什么停（作者 2026-10-19 确认）：**成本太高**。4B 完成了 SFT（第 6 章），之后没有继续 RL 和全量评测。
- 素材：`result/round10/conclusion.md`、`result/round10/harbor-*/`、§3。
- 如果重来（作者 2026-10-19 回答）：大概会直接沿用现有方法，只是预算会更多。

### 第 13 章 工程纪律：让这个项目可复盘的那些规矩

- AGENTS.md 的规则：每轮记录、失败也留档、固定 revision/manifest/seed、先文档后代码、目录 README。
- 资源回收：每次结束都要 VERIFIED；提交前两次核对；不把 STOP 当删除。
- 多端 checkpoint（本地 / 云端 / Drive）与断点续跑。Drive 上分两处：Colab 阶段（round1–2）的数据与 SFT 材料一处，round10 的评测报告和 checkpoint 一处（书里只写"Drive 项目目录"，不写文件夹 ID）。
- 素材：`AGENTS.md`、`docs/apexo/10_agent_platform_plan.md`、各轮 `ledger*.json`、`inventory*.txt`。

### 附录

- A. 时间线与耗时总表（来自 `docs/apexo/11_timeline.md`）。
- B. 坑位速查：R1–R49 里真正踩到的那些，按"现象 → 原因 → 解决"一行一条。
- C. 复现指南：环境、依赖版本、数据 revision、命令（不含任何凭据或项目编号）。
- D. 术语表。
- E. 致谢与第三方许可（文件名 `chapters/99_credits.md`；外部数据集与 harness 的 ID 只能出现在这里）：公开数据集（ODC-BY 1.0）、评测 harness（Apache-2.0）、Harbor / Terminal-Bench 2.0、Qwen 基座模型；按源仓库 `THIRD_PARTY.md` 署名。

## 已知记录缺口

无待作者回答的问题（截至 2026-10-19）。

已解决（作者 2026-10-19 回答）：
- 实际花费：约 200 美元，全部是 Google Cloud。
- 4B 停止原因：成本太高。
- 方法归属：Apexo，作者自行设计。
- 门禁阈值：Apexo 阈值，见第 10 章。
- 源仓库引用方式：保持私有，用纯文本写路径和提交号。
- 为什么去掉 Daytona：GCP 上已有能承载沙箱的 VM。
- 为什么选 2B：足够小。
- 如果重来：沿用现有方法，加大预算。

记录说明（不是待问问题）：RL 评测 Attempt 1 的原始台账和日志已被 Attempt 2 覆盖。Drive 上只留下一份较早的编排状态快照（316 trials 时点），已收进源仓库 `result/round10/harbor-rl-2b/attempt1-orchestrator-status-drive.json`。书里照此写，不要补细节。

## 交付批次

| 批次 | 章节 |
| --- | --- |
| 1 | 第 0、1 章 |
| 2 | 第 2、3、4 章 |
| 3 | 第 5、6 章 |
| 4 | 第 7、8、9 章 |
| 5 | 第 10、11 章 |
| 6 | 第 12、13 章与附录 |
