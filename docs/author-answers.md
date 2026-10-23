# Q1–Q10 作者答案（Coding 返修）

每题固定三个小节：记录依据、常见做法与出处、结论。标签含义：[记录事实] 来自固定提交；[读者运行所得] 来自公开脚本或本仓库可复算数据；[作者分析] 是取舍解释。公开稿只保留可公开复核的路径、提交号和来源链接。

## Q1：模型结构

### 记录依据
[记录事实] `固定实验提交 ca57456/train/model_utils.py` 将官方 Qwen3.5 文本模型加载为 Qwen3_5ForCausalLM；官方视觉语言权重、文本模型和 vLLM 运行时使用不同命名空间，导出与同步分别映射。LoRA 目标包含注意力、GDN 和 MLP 投影，lm_head 由 `_assert_head_frozen` 检查为冻结。

### 常见做法与出处
[记录事实] LoRA 把低秩增量挂到选定线性层并通常冻结主权重，见 Hu 等 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)；实现侧配置语义见 [PEFT LoRA 文档](https://huggingface.co/docs/peft/conceptual_guides/lora)。

### 结论
【作者事后分析】 本项目选择文本因果加载路径是为了保持 Base→SFT→RL 的可审计边界。公开随机两层示例只能验证形状契约，不能替代第 6 章与 Q4 对固定 revision 的参数计数。

## Q2：r=64、alpha=64、dropout=0

### 记录依据
[记录事实] `固定实验提交 ca57456/configs/sft/qwen35_2b_lora.yaml` 与 `configs/rl/qwen35_2b.yaml` 都写明 r=64、alpha=64、dropout=0.0；`train/model_utils.py` 用同样默认值构造 PEFT 配置。

### 常见做法与出处
[记录事实] LoRA 论文将缩放写作 α/r（见 [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) §4.1）；α=r 时缩放因子为 1。推理与训练配方侧的默认讨论见 [QLoRA](https://arxiv.org/abs/2305.14314) 与 Hugging Face [LoRA 概念指南](https://huggingface.co/docs/peft/conceptual_guides/lora)。

### 结论
【作者事后分析】 α 与 r 相等使缩放为 1；dropout=0 便于小样本 SFT→RL warm-start 的确定性衔接。该解释是事后分析，不是当时日志里的决策原文。

## Q3：冻结 embedding、norm、head

### 记录依据
[记录事实] `固定实验提交 ca57456/train/model_utils.py` 的目标模块列表不包含 embedding/norm，且 `_assert_head_frozen` 会拒绝可训练 lm_head；Qwen3.5 在绑定 embedding/lm_head 时仍保留冻结 head 约束。

### 常见做法与出处
[记录事实] 参数高效微调通常只训练适配器而冻结词表与输出头，以控制显存并降低词表漂移；LoRA 原论文与 [PEFT 冻结策略说明](https://huggingface.co/docs/peft/developer_guides/lora) 都把“主权重冻结、增量可训练”作为默认合同。

### 结论
【作者事后分析】 冻结共享 embedding、归一化和 lm_head 可降低显存与漂移风险；代价是不能用全参数更新修复词表或归一化统计。

## Q4：总参数与可训练参数

### 记录依据
[记录事实] 两份 YAML 固定官方 `Qwen/Qwen3.5-2B-Base` revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`，LoRA r=64、α=64、dropout=0.0，目标模块 12 类：`q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj,gate_proj,up_proj,down_proj`。本仓库快照 `data/q4_lora_2b_parameters.json` 用 transformers==5.6.0 的 `Qwen3_5ForCausalLM(text_config)` 在 meta 设备上实例化，再套 peft==0.18.1 LoRA，对匹配到的 186 个 Linear 叶按 `r×(in+out)` 求和，并与 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 对齐。

### 常见做法与出处
[记录事实] 可训练参数应按适配器叶子计数，而不是只报“12 个 target 名字”；形状合同见 LoRA 论文与 [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)。本项目第 6 章用同公式在随机两层夹具上核验形状，Q4 则对官方 config 做全量计数。

### 结论
【读者运行所得】 固定 revision 下：`base_unique_parameters=1881825088`，`lora_trainable_parameters=67276800`，`wrapped_unique_parameters=1949101888`，`n_target_instances=186`，可训练占比约 `0.0345168`。例如 `q_proj` 在本架构为 `2048→4096`（含 attn output gate），单层 LoRA 增量 `64×(2048+4096)=393216`；全模型求和即上列整数。数字以 `data/q4_lora_2b_parameters.json` 为准，测试会在改坏 JSON 或正文后重新绑定并失败。

## Q5：pass@1 与 valid-call rate

### 记录依据
[记录事实] `固定实验提交 ca57456/evaluation/run_eval.py` 的 `summarize` 先按 task 聚合 pass@1；valid-call rate 由每个 trial 的 timing.json 中 `format_error` 步数计算，缺失或无效 timing 进入失败门禁。

### 常见做法与出处
[记录事实] pass@k 的无偏估计来自 Chen 等 [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)；工具调用有效率是步骤级协议指标，应与任务等权成功率分开报告。

### 结论
【作者事后分析】 pass@1 应按任务等权汇总，valid-call rate 应按步骤口径汇总；报告必须分别给出分母和缺失 timing 的处理。

## Q6：paired bootstrap

### 记录依据
[记录事实] `固定实验提交 ca57456/evaluation/run_eval.py` 的 `paired_bootstrap` 取任务交集，seed=0，默认 iters=2000，返回均值和 2.5%/97.5% 分位点。公开评测快照：`result/round10/harbor-base/baseline.json`、`result/round10/harbor-sft-2b/sft-gate-apexo.json`、`result/round10/harbor-rl-2b/rl-gate-apexo.json`（均在 ca57456）。逐任务 pass@1 与复算结果写入 `data/q6_paired_bootstrap.json`。

### 常见做法与出处
[记录事实] 成对 bootstrap 用任务级差分估计均值增益的置信区间，是 A/B 评测的标准做法；Efron & Tibshirani 的 bootstrap 框架见经典综述，NLP/代码评测中的成对用法与 pass@k 报告常见于 [Codex 评测设定](https://arxiv.org/abs/2107.03374) 及后续 agent 评测门禁。本项目门禁要求 SFT 相对 Base 的 95% CI 下界 > 0，且均值增益达到阈值。

### 结论
【读者运行所得】 对 89 个交集任务、iters=2000、seed=0：SFT−Base 均值增益 `0.011235955056179775`，95% CI `[0.0022471910112359553, 0.02471910112359551]`；RL−SFT 均值增益 `-0.0022471910112359553`，95% CI `[-0.011235955056179775, 0.006741573033707866]`。这与闸门文件内嵌的 `gate.delta_pass_at_1` 一致。代价是：增益小、区间贴零时，换阈值（例如旧的 +0.02）会改变 PASSED/FAILED，所以报告必须同时写清阈值版本。

## Q7：KL 0.2 监控

### 记录依据
[记录事实] `固定实验提交 ca57456/configs/rl/qwen35_2b.yaml` 设置 `kl_monitor_every`（续训覆盖为每步）；`evaluation/run_eval.py` 默认 `kl_max=0.2`。`data/q7_q9_rl_metrics.json` 从 `result/round10/rl-2b/continue-final/metrics.jsonl` 抽出：step 5 的 `train/kl_k3_vs_base=0.1978531239748372`，全窗最大同为该值。

### 常见做法与出处
[记录事实] PPO 用 KL 惩罚或监控防止策略漂移，见 Schulman 等 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)；GRPO/群体相对方法同样把相对 Base/参考策略的散度当作训练健康指标，见 [DeepSeekMath GRPO](https://arxiv.org/abs/2402.03300)。本项目 beta=0，KL 只监控不进 loss。

### 结论
【作者事后分析】 0.2 是项目门禁阈值；step 5 的 0.1978531239748372 未越过它，但单点不能证明“已校准”。完整曲线（见同 JSON 的 `kl_values`）和修订记录才足以评价阈值松紧。

## Q8：熵与格式告警

### 记录依据
[记录事实] `evaluation/run_eval.py::rl_training_checks` 的硬门禁是：`train/entropy` 末值不得超过首值的 3 倍；最近至多 10 条训练行的 `done/no_tool_call ÷ rollout/samples` 均值不得超过 `0.10`。`data/q8_entropy_format.json` 对 continue-final 复算：entropy 序列首 `0.22409780968993934`、末 `0.3875395980056593`，比值约 `1.729`；`recent_no_tool_call_frac=0.08229166666666667`。固定提交没有单独的“2×/3× 运营告警工单”，但代码里的 3× 与 0.10 就是可执行阈值。

### 常见做法与出处
[记录事实] 策略熵崩溃/爆炸是 RLHF/RLAIF 常见失效模式，PPO 系工作强调监控熵与 KL（[Schulman et al., 2017](https://arxiv.org/abs/1707.06347)）；工具调用智能体还需单独监控“无工具调用结束”比例，以免格式塌陷被平均奖励掩盖（与本项目 `no_tool_call_max=0.10` 同族）。

### 结论
【读者运行所得】 本窗未触发熵 3× 爆炸，也未越过 0.10 格式塌陷线；step 3 单步 no-tool 比例曾到 `0.2`，说明均值合格仍可能有尖峰。预警线可以低于硬门槛，但公开稿只陈述代码里真实存在的阈值与复算值，不编造缺失的告警工单。

## Q9：末十步格式塌陷

### 记录依据
[记录事实] `data/q7_q9_rl_metrics.json` 来自 `固定实验提交 ca57456/result/round10/rl-2b/continue-final/metrics.jsonl` 的 8 个训练 step。按 `done/no_tool_call ÷ rollout/samples`，step 1–8 为 0.0625、0.125、0.20、0.0416667、0.0625、0、0.125、0.0416667；平均 `0.0822917`。脚本的最近十步窗口会使用实际存在的 8 行，不把 eval 行当训练行。

### 常见做法与出处
[记录事实] 格式/协议塌陷应在滚动窗口上聚合，而不是只看最后一步；与 Q5 的步骤级 valid-call、Q8 的硬门禁同一度量族。相关评测实践见 Chen 等 [pass@k 设定](https://arxiv.org/abs/2107.03374) 对“分母与聚合口径必须写清”的要求。

### 结论
【读者运行所得】 复算平均值 0.0822917 低于 0.10 门槛，但仍需结合 entropy、truncated_frac、format_errors 与全程分布。窗口均值合格不能单独证明无格式塌陷。

## Q10：Base 与 Instruct

### 记录依据
[记录事实] `固定实验提交 ca57456/AGENTS.md` 与两份 YAML 固定 Qwen 官方预训练 Base，并要求 SFT adapter 作为 RL warm-start；直接从 Base 做 RL 只允许显式 `allow_raw_base=true`，默认关闭。

### 常见做法与出处
[记录事实] 从 Base 再跑可控 SFT/RL，便于审计增量；直接从 Instruct/Chat 出发则混入厂商未公开的对齐先验。Qwen 官方模型卡区分 Base 与 Instruct 变体（见 [Qwen3.5-2B-Base 模型卡](https://huggingface.co/Qwen/Qwen3.5-2B-Base)）；后训练综述中的“起点模型选择”讨论见 [InstructGPT](https://arxiv.org/abs/2203.02155) 与后续 RLHF 综述实践。

### 结论
【作者事后分析】 本项目选 Base 是为了隔离 SFT/RL 增量并保留可审计起点；Instruct 更适合直接推理，但会混入未记录的后训练先验。这是本项目范围决策，不是所有应用的普遍结论。

## 复现

```bash
# Q4（需 transformers==5.6.0 + peft；meta 设备，不下载权重）
python examples/ch06_official_config_optional.py  # 可选：仅下载 config
# 本仓库已提交 data/q4_lora_2b_parameters.json，测试直接绑定该文件。

# Q6/Q7/Q8/Q9：测试从 data/q6_paired_bootstrap.json、data/q7_q9_rl_metrics.json、
# data/q8_entropy_format.json 重算并对照正文；故意改坏 JSON 或正文数字必须失败。
python3 -m pytest -q tests/test_author_answers_provenance.py
python3 scripts/check_book.py
```
