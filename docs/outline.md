# 《后训练从 0 到 1》详细大纲（v0，待写作）

作者：Hawk Wu

本书根据作者的真实项目 `qwen-posttrain`（以下简称"源仓库"，当前为私有仓库）整理：在官方 `Qwen/Qwen3.5-2B-Base` / `Qwen3.5-4B-Base` 上，按 TMAX 方法做终端智能体（Terminal Agent）后训练，从 2026-10-10 第一次提交到 2026-10-19 结项。全书按"先打基础，再按时间讲真实过程"组织：每一步写清楚**做了什么、为什么这么做、遇到了什么坑、怎么解决、花了多久、之后怎么调整**。

## 写作规则

1. **每个事实都要能追溯**到源仓库的文件、日志、指标或提交（写明路径或 commit）。数字（loss、显存、耗时、pass@k）照抄原始记录，不四舍五入成别的意思。
2. **记录里没有的不编**。原因、心路、费用等如果找不到，就在章末"待作者确认"里列问题，不要替作者补动机。
3. **如实写结果**，包括失败和不好看的数字：严格门禁两次判 FAILED、RL 后 pass@1 比 SFT 低，都要写，并说明作者为什么仍然放行。
4. 公开书稿里**不出现**云项目编号、bucket 名、Drive 文件夹 ID、服务器主机名、邮箱、密钥或任何凭据；用"项目 bucket""训练服务器"等代称。
5. 风格：中文、面向能写 Python 但没做过后训练的读者；概念先用直觉解释，再给最小代码；源仓库代码只摘关键片段并注明出处。
6. 时间：原始日志多为 UTC，正文统一写成"日期 + 时长"，需要具体时刻时注明时区。

## 章节与素材映射

路径都相对于源仓库根目录。"§x" 指 `docs/tmax/00_progress.md` 的小节；"R" 编号指 `docs/tmax/06_risks_and_decisions.md` 的风险登记；时间线见 `docs/tmax/11_timeline.md`（源仓库 PR #6）。

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
- 本项目路线：TMAX 数据 → LoRA SFT 预热 → outcome-only DPPO → Harbor + Terminal-Bench 2.0 评测；相对 TMAX 的偏差（单卡、LoRA、lr 等）。
- 素材：`AGENTS.md`、`docs/tmax/01_architecture.md`、`docs/tmax/09_custom_training_design.md`（§1.3 偏差表）、`configs/rl/qwen35_2b.yaml` 注释、`docs/tmax/README.md` 流程图。
- 待作者确认：为什么选 TMAX、为什么选 Qwen3.5 Base 而不是 Instruct、为什么是 2B/4B。

### 第 3 章 第一版为什么不能用：从 v1 到 v2（10-10）

- v1 审核发现的 P0/P1 问题逐条讲：RL 训练器是空壳（硬编码 dummy response）、依赖还是 Qwen2.5、工具调用格式不是 Qwen3.5 原生、SFT 没有 collator、评测无结果时写假的 PASSED……每条讲"坑在哪 → 怎么发现 → v2 怎么改"。
- 教训：先审核再训练；"能跑"不等于"在训练"。
- 素材：§1.2 表、§1.3、§6.1–§6.4；提交 `047c932`、`fbf621e`；PR #1。

### 第 4 章 数据：把 1.5 万个任务冻结下来（10-12，round1）

- 数据源：TMAX 15K RL 任务包与 tmax-sft 轨迹；固定 revision、seed 42、留出 200 条；manifest + SHA-256。
- 数量与分布：rl_train 14,401 / heldout 200 / SFT 5,795 条；14,490 个不同镜像；SFT token 长度分位数与 `max_seq_len=24576` 的取舍；数据瑕疵（缺思考前缀、无工具调用轮、残留 `</think>`、非 bash 工具名）怎么处理。
- 过程：Colab CPU 上约 20 分钟跑完；D1/D2 校验。
- 素材：`docs/tmax/03_data_plan.md`、§2、`result/round1/`、`scripts/data/`、`tests/test_dataset.py`、`tests/test_manifest_and_eval.py`。

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
- 素材：`result/round3/`–`result/round9/` 各 README、`docs/tmax/07_gcp_colab_a100_training_plan.md`、`docs/tmax/10_agent_platform_plan.md`、`result/round10/scripts/preflight_fit.py`、`result/round8/costs-and-retention.json`。
- 待作者确认：实际花费；为什么放弃 Daytona。

### 第 8 章 RL 原理与实现：DPPO 怎么在一张卡上跑

- 一步 RL 的完整流程：抽 4 个任务 × 每个 8 个样本 → vLLM 多轮 rollout（训练器离卡）→ 沙箱里执行 → verifier 0/1 奖励 → centered advantage → DPPO（binary TV 0.1）更新 LoRA → 权重推回 vLLM。
- 关键设计：零方差组过滤与有界 active sampling、截断样本 mask、`sync/logprob_abs_diff` 同步检查、k3 KL 只监控不惩罚（beta=0）、fp32 lm_head、分块 logprob 防 248K 词表 OOM、vLLM sleep/wake。
- 素材：`train/rl_trainer.py`、`train/rollout.py`、`train/losses.py`、`train/agent_env.py`、`configs/rl/qwen35_2b.yaml`、`docs/tmax/09_custom_training_design.md`、`docs/tmax/08_execution_contract.md`、`tests/test_rl_step.py`、`tests/test_losses.py`、提交 `1037af0`、`a5a8046`。

### 第 9 章 沙箱：让模型的命令只在该在的地方跑

- 为什么模型命令绝不能在宿主机裸跑；GPU worker 不跑 Docker；临时 CPU VM 跑原任务容器，经 IAP SSH 调用；VM 无服务账户、容器拿不到凭据（R44）。
- round5 原生 Shell Sandbox 实测（能力和限制），round9 原任务 verifier 探针（只证明链路通，不证明正确解法得 1 分）。
- 素材：`train/sandbox.py`、`train/remote_sandbox.py`、`scripts/agent_platform/cpu_startup.sh`、`result/round5/`、`result/round9/`、R40、R44。

### 第 10 章 评测：Harbor 与 Terminal-Bench 2.0 的长跑（10-14 ~ 10-17）

- 指标：89 个任务 × 5 次 = 445 trials；pass@1/3/5、valid_tool_call_rate；门禁怎么定。
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
- 严格门禁两次 FAILED、作者人工放行（`sft-user-approved.json`、`rl-user-approved.json`）——讲清楚放行依据和它不能说明什么。
- 只做了 8 步 RL、2B 单卡，结论边界；没有复现 TMAX 论文数字。
- 素材：`result/round10/conclusion.md`、`result/round10/harbor-*/`、§3。
- 待作者确认：放行时的考虑；如果重来，最想改什么。

### 第 13 章 工程纪律：让这个项目可复盘的那些规矩

- AGENTS.md 的规则：每轮记录、失败也留档、固定 revision/manifest/seed、先文档后代码、目录 README。
- 资源回收：每次结束都要 VERIFIED；提交前两次核对；不把 STOP 当删除。
- 多端 checkpoint（本地 / 云端 / Drive）与断点续跑。
- 素材：`AGENTS.md`、`docs/tmax/10_agent_platform_plan.md`、各轮 `ledger*.json`、`inventory*.txt`。

### 附录

- A. 时间线与耗时总表（来自 `docs/tmax/11_timeline.md`）。
- B. 坑位速查：R1–R49 里真正踩到的那些，按"现象 → 原因 → 解决"一行一条。
- C. 复现指南：环境、依赖版本、数据 revision、命令（不含任何凭据或项目编号）。
- D. 术语表。

## 已知记录缺口（写到相关章节时问作者）

1. 实际花费：仓库没有账单金额。
2. 为什么停掉 4B：只记录了"按用户指令"。
3. 为什么选 TMAX、Qwen3.5 Base、2B/4B；为什么放弃 Daytona。
4. RL 评测 Attempt 1 自己的台账和日志没有进仓库（两份文件内容相同，都是 Attempt 2）。
5. 两次人工放行时的具体考虑。
6. 书里能否链接源仓库：源仓库目前是私有的，公开读者打不开链接。

## 交付批次

| 批次 | 章节 |
| --- | --- |
| 1 | 第 0、1 章 |
| 2 | 第 2、3、4 章 |
| 3 | 第 5、6 章 |
| 4 | 第 7、8、9 章 |
| 5 | 第 10、11 章 |
| 6 | 第 12、13 章与附录 |
