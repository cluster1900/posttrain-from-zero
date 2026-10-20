# Q1–Q10 作者答案（返修版）

每题固定三个小节：记录依据、常见做法与出处、结论。标签含义：[记录事实] 来自固定提交；[读者运行所得] 来自公开脚本或本文件的可复算小样例；[作者分析] 是取舍解释。公开稿不包含私有 URL、账号、主机名、云项目或凭据。

## Q1：模型结构

### 记录依据
[记录事实] `cluster1900/qwen-posttrain@ca57456/train/model_utils.py` 将官方 Qwen3.5 文本模型加载为 Qwen3_5ForCausalLM；官方视觉语言权重、文本模型和 vLLM 运行时使用不同命名空间，导出与同步分别映射。LoRA 目标包含注意力、GDN 和 MLP 投影，lm_head 由 `_assert_head_frozen` 检查为冻结。

### 常见做法与出处
[记录事实] PEFT LoRA 将低秩增量挂到选定线性层，主模型权重通常冻结；见 LoRA 论文 https://arxiv.org/abs/2106.09685 与 PEFT 文档 https://huggingface.co/docs/peft 。

### 结论
[作者分析] 本项目选择文本因果加载路径是为了保持 Base→SFT→RL 的可审计边界。公开随机两层示例只能验证形状，不能当历史 2B 参数测量。

## Q2：r=64、alpha=64、dropout=0

### 记录依据
[记录事实] `@ca57456/configs/sft/qwen35_2b_lora.yaml` 和 `configs/rl/qwen35_2b.yaml` 都写明 r=64、alpha=64、dropout=0.0；`train/model_utils.py` 对同样默认值调用 PEFT。

### 常见做法与出处
[记录事实] LoRA 的缩放通常使用 alpha/r，dropout 是可选正则项；PEFT 的配置语义与 LoRA 原论文一致。常见工程做法是先固定 rank 和缩放，再用验证集调 dropout，而不是同时改变三者。

### 结论
[作者分析] alpha 与 r 相等使缩放因子为 1；dropout=0 便于小样本 SFT→RL warm-start 的确定性衔接。该解释是事后分析，不伪称为当时日志中的决策记录。

## Q3：冻结 embedding、norm、head

### 记录依据
[记录事实] `@ca57456/train/model_utils.py` 的目标模块列表不包含 embedding/norm，且 `_assert_head_frozen` 会拒绝可训练 lm_head；Qwen3.5 的 lm_head 与 embedding 采用绑定布局时，项目保留冻结 head 约束。

### 常见做法与出处
[记录事实] 参数高效微调通常冻结基础模型，只训练适配器；PEFT 文档和 LoRA 论文都把冻结主权重作为基本设定。

### 结论
[作者分析] 冻结共享 embedding、归一化和 lm_head 可降低显存与漂移风险；代价是不能用全参数更新修复词表或归一化统计。

## Q4：总参数与可训练参数

### 记录依据
[记录事实] 两份 YAML 固定官方 Qwen3.5-2B-Base revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` 与 LoRA r=64；`model_utils.py` 提供 `lora_parameters`，按 `requires_grad` 返回可训练参数。

### 常见做法与出处
[记录事实] 业界报告参数量时区分总参数、可训练参数和激活参数；PEFT 的标准核对方式是对 `named_parameters()` 的 `requires_grad` 求和，而不是从模型名猜数字。

### 结论
[读者运行所得] 在固定 Base revision 上运行 `sum(p.numel() for p in model.parameters())` 与 `sum(p.numel() for p in model.parameters() if p.requires_grad)` 即得到本机实际总量/LoRA 量；两者必须连同 revision、dtype、target_modules 一起记录。本 PR 不把随机示例或缺失日志写成历史绝对数。

## Q5：pass@1 与 valid-call rate

### 记录依据
[记录事实] `@ca57456/evaluation/run_eval.py` 的 `summarize` 先按 task 聚合 pass@1；valid-call rate 由每个 trial 的 timing.json 中 format_error 步数计算，缺失或无效 timing 进入失败门禁。

### 常见做法与出处
[记录事实] pass@k 的无偏估计来自 Chen 等的 pass@k 设定 https://arxiv.org/abs/2107.03374；工具调用有效率是步骤级协议指标，不能与任务等权成功率混合。

### 结论
[作者分析] pass@1 应按任务等权汇总，valid-call rate 应按步骤口径汇总；报告必须分别给出分母和缺失 timing 的处理。

## Q6：paired bootstrap

### 记录依据
[记录事实] `@ca57456/evaluation/run_eval.py` 的 `paired_bootstrap` 取任务交集，seed=0，默认 iters=2000，返回均值和 2.5%/97.5% 分位点。

### 常见做法与出处
[记录事实] 配对重采样保持同一任务在当前报告与基线中的对应关系，避免把不同任务集合当独立样本；这是评估差异时的常见 bootstrap 设计。

### 结论
[读者运行所得] 用两个任务差值 [0.10, 0.20] 调用 `paired_bootstrap` 可得到可复算的均值和 CI；正式结果必须输入逐任务报告和固定 seed/iters。本仓库未保存公开逐任务 bootstrap 快照，因此本 PR 不伪造正式 CI 端点。

## Q7：KL 0.2 监控

### 记录依据
[记录事实] `@ca57456/configs/rl/qwen35_2b.yaml` 设置 `kl_monitor_every=10`；`evaluation/run_eval.py` 默认 `kl_max=0.2`。固定文件 `result/round10/rl-2b/continue-final/metrics.jsonl` 的 step 5 为 `train/kl_k3_vs_base=0.1978531239748372`。

### 常见做法与出处
[记录事实] PPO/GRPO 类训练常同时监控策略漂移、熵和格式退化；KL 监控是门禁信号，不等于训练目标中加入 KL 惩罚。

### 结论
[作者分析] 0.2 是项目门禁阈值；step 5 未越过它，但单点不能证明“已校准”。完整曲线和修订记录才足以评价阈值。

## Q8：熵与格式告警

### 记录依据
[记录事实] `run_eval.py` 只对训练行计算最近窗口的 no-tool-call 比例，并将固定上限写为 0.10；metrics 同时记录 `train/entropy`。

### 常见做法与出处
[记录事实] 熵监控通常用于发现探索塌陷或异常扩散，格式调用率则是任务协议指标；二者应分开画曲线，不能用单一阈值替代。

### 结论
[作者分析] 预警线可以低于硬门槛以便提前观察趋势，但当前固定提交没有独立的“2×/3×告警决策”记录，所以不把该历史决策写成事实。

## Q9：末十步格式塌陷

### 记录依据
[记录事实] `@ca57456/result/round10/rl-2b/continue-final/metrics.jsonl` 有 8 个训练 step。按 `done/no_tool_call ÷ rollout/samples`，step 1–8 分别为 0.0625、0.125、0.20、0.0416667、0.0625、0、0.125、0.0416667；8 步平均为 0.0822917。当前脚本的最近十步窗口会使用实际存在的 8 行，不把 eval 行当训练行。

### 常见做法与出处
[记录事实] 训练日志用固定窗口观察近期退化，同时保留全程曲线；窗口长度必须写明，不能把最后窗口替换成全程平均。

### 结论
[读者运行所得] 运行同一 JSONL 的小脚本即可复算上述 8 个比例和平均值；0.0822917 低于 0.10 门槛，但仍需结合 entropy、truncated_frac、format_errors 和全程分布。窗口结果不能单独证明无格式塌陷。

## Q10：Base 与 Instruct

### 记录依据
[记录事实] `@ca57456/AGENTS.md` 与两份 YAML 固定 Qwen 官方预训练 Base，并要求 SFT adapter 作为 RL warm-start；直接从 Base 做 RL 只允许显式 `allow_raw_base=true`，默认关闭。

### 常见做法与出处
[记录事实] Base→SFT→RL 与直接使用 Instruct 是不同实验起点；模型卡、训练论文和评测报告都要求注明起点与 revision，不能把不同起点的分数直接比较。

### 结论
[作者分析] 本项目选 Base 是为了隔离 SFT/RL 增量并保留可审计起点；Instruct 更适合直接推理但会混入未记录的后训练先验。这是本项目范围决策，不是所有应用的普遍结论。

## 复现与当前边界

每题的固定提交、路径、字段和运行命令都应随 PR 保存。当前仍缺：逐任务正式评测快照、Q6 正式 bootstrap 输出、Q8 历史告警决策记录。缺口保持显式，不用推断补齐。
