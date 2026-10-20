# Q1–Q10 作者答案（证据分层版）

本文件对应 #35 的 Q1–Q10 返修批次。每条结论先标注证据性质：
- [记录事实] 可在固定提交的源文件中定位。
- [读者运行所得] 只表示公开示例/测试实际运行结果。
- [作者分析] 解释设计取舍，不伪装成历史实验事实。

公共稿件不包含私有仓库链接、账号、主机名、云项目或凭据。私有取证仅记录仓库、路径和提交号。

## Q1：模型结构
[记录事实] `cluster1900/qwen-posttrain@ca57456/train/model_utils.py` 将官方 Qwen3.5 文本因果模型加载为 Qwen3_5ForCausalLM；官方视觉语言权重、文本模型和 vLLM 运行时使用不同命名空间，导出与同步分别映射。
[记录事实] `model_utils.py` 的 `apply_lora` 默认 r=64、alpha=64、dropout=0，并在 LoRA 后检查 lm_head 冻结。
[读者运行所得] 公开仓库的随机两层示例只能验证张量形状与 reader 实验，不能代表历史 2B 参数量。

## Q2：为什么 r=64、alpha=64、dropout=0
[记录事实] `@ca57456/configs/sft/qwen35_2b_lora.yaml` 与 `configs/rl/qwen35_2b.yaml` 均使用 r=64、alpha=64、dropout=0.0。
[作者分析] 这是在单卡显存预算下提高可训练子空间、同时保持 LoRA 缩放为 1 的工程选择；dropout=0 便于小样本 SFT→RL warm-start 的确定性衔接。该解释是事后分析，不是日志中的历史决策记录。

## Q3：为什么冻结 embedding/norm/head
[记录事实] `@ca57456/train/model_utils.py` 只把 LoRA 目标模块交给 PEFT，并通过 `_assert_head_frozen` 拒绝可训练 lm_head；配置目标包含注意力、GDN 和 MLP 投影，不包含 embedding/norm。
[作者分析] 冻结共享 embedding、归一化和 lm_head 可降低显存与漂移风险；代价是不能用全参数更新修复词表或归一化统计。

## Q4：总参数与可训练参数
[记录事实] 配置固定官方 Qwen3.5-2B-Base revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`，LoRA r=64。公开稿没有一份历史 trainable-params 日志，因此不提供未经记录的绝对数。
[作者分析] 可训练参数应以 PEFT 实例的 `requires_grad` 计数，不能拿随机示例的参数量冒充 2B 项目实测。

## Q5：pass@1 与 valid-call rate
[记录事实] `@ca57456/evaluation/run_eval.py` 按任务汇总 pass@1，并从每个 trial 的 timing.json 统计格式错误步数；缺失/无效 timing 会进入失败门禁。
[作者分析] pass@1 是任务等权平均，valid-call rate 是步骤级有效调用比例，二者不能混为 token 或 trial 加权的单一指标。

## Q6：paired bootstrap
[记录事实] `@ca57456/evaluation/run_eval.py` 的 `paired_bootstrap` 使用确定性 seed=0，默认 2000 次重采样，按任务交集比较当前报告与基线。
[记录事实] 公共稿未保存逐任务 bootstrap 快照，因此不能在此文件写出新的端点或 CI 数字；需要对应报告和每任务结果才能重算。

## Q7：KL 0.2 校准
[记录事实] `@ca57456/configs/rl/qwen35_2b.yaml` 将 `kl_monitor_every=10`，门禁脚本默认 `kl_max=0.2`。在 `@ca57456/result/round10/rl-2b/continue-final/metrics.jsonl` 的 step 5，`train/kl_k3_vs_base=0.1978531239748372`，未超过该阈值。
[作者分析] 0.2 是门禁阈值，不等于已完成系统校准；需要完整运行的 KL 曲线与阈值修订记录才能评价校准质量。

## Q8：熵告警线与门槛
[记录事实] `@ca57456/evaluation/run_eval.py` 将格式塌陷的固定门槛写为最近训练行 no-tool-call 平均比例不高于 0.10；metrics 同时记录 entropy。
[作者分析] 告警线可低于硬门槛以便提前观察趋势，但本仓库没有一份单独的“2×告警/3×门槛”历史决策记录，因此不把该解释写成当时的事实。

## Q9：末十步格式塌陷
[记录事实] `@ca57456/result/round10/rl-2b/continue-final/metrics.jsonl` 提供每步 `done/no_tool_call` 与 `rollout/samples`，且 `run_eval.py` 只按训练行计算最近十步比例，避免把 eval 行当作 0% 工具调用。
[记录事实] 该文件显示 step 1–8 的训练行均可定位；本文件不新增“全程 vs 末十步”的汇总数字，避免手工重算后误标原始记录。
[作者分析] 末十步窗口用于发现近期格式退化，不能代替全程分布。

## Q10：Base 与 Instruct 的选择
[记录事实] `@ca57456/AGENTS.md` 和两份 YAML 均固定 Qwen 官方预训练 Base，并要求 SFT adapter 作为 RL warm-start；直接从 Base 做 RL 只允许显式 `allow_raw_base=true`。
[作者分析] Base 提供从零可审计的训练起点，Instruct 更适合直接推理但会混入未记录的后训练先验；因此本项目选择 Base 是范围与可复现性决策，不是对所有应用的普遍结论。

## 复现与缺口

复现前先取得固定提交的源副本，再按路径核对字段；不要把私有文件复制进公开稿。当前仍缺：逐任务评测原始结果和 bootstrap 快照、完整 Q9 窗口汇总、Q8 历史告警决策记录。缺口保持显式，后续补证必须追加路径+提交号和测试输出。
