# 第 9 章 RL 原理与实现：DPPO 怎么在一张卡上跑

## 动机

SFT 告诉模型哪些续写出现在成功轨迹中，结果奖励 RL 则把模型自己产生的行动放回环境，再用测试结果更新策略。难点不是再写一个 loss 名字，而是保证这次生成、这份奖励、这条 log-probability 和将要更新的权重说的是同一件事。

Apexo 的这一实现采用 DPPO 的二元总变差近似与组内中心化优势，继承公开方法而非发明新算法。DPPO 的公开出处是 Penghui Qi 等的 [Rethinking the Trust Region in LLM Reinforcement Learning](https://arxiv.org/abs/2602.04879)。本章按项目源实现解释其选用分支，不将论文所有变体都说成项目已经实现，也不借论文上的结果替本项目宣称增益。

主要源文件是 `train/rl_trainer.py`、`train/losses.py`、`train/rollout.py` 与 `train/agent_env.py`，结项提交 `ca57456`。本章讲机制和离线 CPU 核验；真实训练奖励、总耗时和最终评测留给后续章节。

![图 9-1 一步结果奖励 RL 的因果链；每条轨迹的行为策略概率随生成一起保存](images/ch09_loop.svg)

## 配置

### 9.1 四个问题，各采八条，不等于32个容器同时运行

归档的 `configs/rl/qwen35_2b.yaml` 从同一官方基座上的 SFT adapter 起步，默认拒绝无 adapter 的裸基座训练；`allow_raw_base` 仅是显式诊断开关。4B 有对应独立配置，不能据2B长度替4B背书。

| 项目 | 2B 归档值 | 意义 |
| --- | --- | --- |
| 每步初始问题数 P | 4 | 四组不同任务 |
| 每问题样本数 G | 8 | 同一任务的八次独立轨迹 |
| 初始采样总数 | 32 | 不含后续补采 |
| 活跃环境上限 | 8 | 用滚动准入完成所有样本 |
| 温度 / top-p | 1.0 / 1.0 | 与行为概率合同一致，源配置校验要求如此 |
| 总序列 / 单轮生成 / 提示上限 | 24,576 / 8,192 / 4,096 | 是上限，不是每条实际长度 |
| 最大行动轮数 | 50 | 达到边界的轨迹按截断合同处理 |
| 优势 / DPPO 阈值 | centered / 0.1 | 不除组标准差，使用 binary TV |
| 零方差过滤 / 补采 | 开 / 最多3轮补采 | 补采有界，不保证最终凑满 |
| 学习率 / warmup | 1e-5 / 10步 | 后续常数，不是完整模型全参训练配方 |
| 最大梯度范数 / 分块大小 | 1.0 / 1,024 | 分块计算 fp32 head 的 token log-prob |
| KL 监控 | 每10步 | beta=0，不作为额外惩罚项 |
| vLLM 调度初值 | utilization 0.55、max_num_seqs 8 | 可按实测调整的容量参数 |

32 条样本不意味着一组有32条，更不意味着同时运行32个任务容器。实际补采会增加本步轨迹总数，统计时不能永远用32作分母。训练长度配置也不能替代实际长度与显存验收。

### 9.2 奖励先分清故障，再算中心

对同一任务的有效样本，奖励为任务验证给出的0或1。设有效组内有 n 条，中心化优势为 Aᵢ = rᵢ − mean(r)。若八条中两条成功，成功样本的优势为0.75，失败样本为−0.25。这里没有除标准差，也没有 value model。

基础设施错误先从组内移除，不把它当作模型失败的0奖励。有效样本少于两条，整组丢弃。有效奖励相同的组没有中心化学习信号，默认过滤；全成功组还会进入已解决集合，避免本步补采继续抽到。

截断则不同：达到长度上限的轨迹仍参与组内奖励均值，但默认不产生梯度。若先把它从均值中删掉，就改了其他样本的优势。最终无生成 token 或优势为0的样本同样不构成训练项。

![图 9-2 故障、零方差、截断各有不同位置，不能合并为一个“失败过滤”](images/ch09_filters.svg)

### 9.3 为什么补采必须有上限

初始四组中若只有一组奖励有变化，训练器会再抽缺少的三组，而不是重新抽四组。补采在同一未更新策略下进行，group 编号继续递增；最多补采3轮，还会在没有未解决任务时提前停止。

源实现中的 useful group 是按组统计判断，不等于保证一定存在非截断 token。极端情况下，组奖励有变化，但所有能用的轨迹都被截断过滤，最终训练项仍可能为空。代码会记录跳过更新，而不是制造一个假的梯度。教学例子 `bounded_requests([1,1,0,0,2])` 返回请求数 `[4,3,2,2]`，第五个输入不会被消费；这些计数是虚构演示，不是项目日志。

### 9.4 DPPO 的掩码不是 PPO 的 ratio clip

令 μ 为生成时行为策略对该采样 token 的概率，π 为当前训练策略概率，比例 ρ=π/μ。项目使用二元事件“采样 token / 其他 token”的总变差：d=|π−μ|。这是二元近似，不是完整词表总变差的精确计算。

只有以下两种情况同时越过阈值时，该 token 才被屏蔽：

- A>0、ρ>1，而且 d>0.1：正优势继续增大已偏离过多的概率。
- A<0、ρ<1，而且 d>0.1：负优势继续减小已偏离过多的概率。

如果更新方向会把概率拉回行为策略，即便 d 已大，仍可保留。阈值比较是严格大于，不是大于等于。实现对概率转换和 log-ratio 做数值截断；因此极端区间的梯度不能用无限精度公式直接替代。

![图 9-3 DPPO 方向掩码；越界还要判断优势与概率移动方向](images/ch09_mask.svg)

本项目 loss 是所有 response token 上 −M·ρ·A 的和，再除以全局 response token 数。M 包含响应位置和 DPPO 保留条件，但分母不改成保留 token 数。如果十个响应 token 中只保留两个，除以2会把本步量级放大5倍，已经是另一个目标。

掩码在 no_grad 下生成；不会对离散“是否保留”求导。未触发 log-ratio 数值截断的区域，关于当前 log-prob 的导数是 −M·ρ·A/N。实际模型还需经 log-softmax 链式求导回各个 logits，不能把这个局部导数误当作某一个模型参数的完整梯度。

源代码的关键分支如下，`train/losses.py` L56–64，提交 `ca57456`，连续原样摘录。它属于项目函数内部，不作为本书的独立可运行片段：

```{.python .source-excerpt}
    with torch.no_grad():
        mu = behavior_logprobs.clamp(-30.0, 0.0).exp()
        pi = new_logprobs.detach().clamp(-30.0, 0.0).exp()
        divergence = torch.where(response_mask, (mu - pi).abs(), torch.zeros_like(mu))
        ratio = (new_logprobs.detach() - behavior_logprobs).clamp(-20.0, 20.0).exp()
        outside = divergence > threshold
        bad = ((advantages > 0) & (ratio > 1.0) & outside) | ((advantages < 0) & (ratio < 1.0) & outside)
        keep = (~bad & response_mask).to(new_logprobs.dtype)
    return keep, divergence
```

### 9.5 单卡不是两份完整运行时一起常驻

生成前，训练器把模型与优化器状态移到 CPU，再唤醒 vLLM 的权重、推送合并后的 LoRA 权重、恢复 KV cache，并清空旧 prefix cache。`run_rollouts` 正常返回或抛错时会进入 finally，依次尝试让 vLLM sleep、把训练器移回 GPU。同步 `_wake_and_sync` 在 try 外；finally 内若 sleep 自身抛错，也会阻止后续迁移。因此这是正常路径与 rollout 异常路径的清理安排，不是同步失败或清理自身失败时完整恢复的保证。这里的串行所有权是容量策略，不是把训练算法换成另一套。

![图 9-4 单卡训练与生成交替占用；同步发生在采样前，旧前缀缓存不能跨策略复用](images/ch09_memory.svg)

权重推送数大于0只是必要条件。生成后的训练器还会对选定轨迹重新计算 log-prob，与生成时保存的行为值比较。`sync/logprob_abs_diff` 是更新前检查；`train/vllm_logprob_abs_diff` 是 loss 计算中按训练项汇总的字段，样本口径和时点不同，不能无条件当同一条曲线。

归档配置的同步中止阈值为0.5：发生过真实权重更新后，差值超过它会抛错；非有限差值也会抛错。0.5 是故障拦截值，不是“理想误差”、不是所有验收场景的阈值，更不表示低于它就证明所有行为分布完全相同。

### 9.6 分块 head 与 KL 监控各解决什么

词表约24.8万项，长序列的完整 fp32 logits 很大。项目缓存冻结的 fp32 lm-head，按1024个位置计算目标 token 的 log-prob，反向时重算分块；避免同时保留完整 `[T,V]` logits，并不是省掉了所有词表计算，也不是把训练 head 换成采样词表。

KL 监控暂时禁用 adapter 得到基座参考概率，计算 `exp(d)−d−1`，其中 d 是经数值截断的 reference log-prob 减 current log-prob。源代码称其 k3 估计器；此处只在少量已有轨迹上做监测，不能不加条件地把有限行为策略样本均值称为当前策略对全空间 KL 的精确或无偏值。它没有乘 beta 加到训练 loss，监测到一个数不代表实际进行了 KL 正则。

## 命令

本书的 CPU 练习直接接收微小概率向量，不加载模型、不启动 vLLM、不连接沙箱。标准 CPU 安装沿用 README 的固定环境。

```bash
python examples/ch09_dppo.py
pytest -q tests/test_ch09_dppo.py
```

下面是新写教学函数的可执行调用，演示一条基础设施错误和一条截断样本如何影响同一组。它不复制私有任务内容：

```python
from examples.ch09_dppo import centered_items
items = centered_items([1, 0, 0, 0],
                       [False, False, False, True],
                       [False, False, True, False])
assert [i for i, _ in items] == [0, 1]
assert abs(items[0][1] - 2/3) < 1e-12
assert abs(items[1][1] + 1/3) < 1e-12
```

配套测试覆盖四种优势/移动方向、严格阈值边界、响应掩码、全局分母、空训练项、非有限输入和有界补采。标量 oracle 独立计算27种概率与优势组合，再对照 PyTorch loss 和梯度。测试通过只证明这些小例子与相应合同一致，不证明真实 GPU 多进程、网络和权重同步都正确。

## 时间线与耗时

本章不是新一次训练报告。按项目记录，10-11 前后先打通基础训练代码，10-12 至10-13的无沙箱短链路暴露并修复同步与容量合同，之后迁移到第 8 章的平台路径。源归档保留的是修订后的实现，不能把它直接说成所有早期运行都已使用。

| 阶段 | 时间口径 | 本章能确认的边界 |
| --- | --- | --- |
| 本地损失/接口测试 | 代码测试，不是 GPU wall time | 可验证优势、mask、梯度与形状 |
| 有界补采代码提交 `1037af0` | 10-12 01:43:55 UTC / 09:43:55 SGT | 代码版本时刻，不是训练起止 |
| 无沙箱短 RL 预演 | 台账日级记录10-12至10-13 | 构造奖励，只检验更新与同步链路 |
| 平台真实 RL 前准备 | 10-13至10-14，见第8章 | 算力和原任务环境逐项验收 |
| 本章 CPU 读者练习 | 运行时自行测量 | 不填写作者历史 GPU 步时 |

不把不同阶段的短链路秒数拼成“正式 RL 吞吐”。正式运行的实际步数、生成时间与奖励分布将在后续训练章节按日志单独列出。

## 坑与修复

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 环境故障被当成0奖励 | 基础设施和模型行为混淆 | 算优势前排除 infra_error，有效组不足则丢弃 | `build_train_items` |
| 截断后其他样本优势变了 | 先删截断样本再算均值 | 截断留在 baseline，但不进入梯度 | `train/rl_trainer.py` L239–262 |
| 低信号任务导致无界补采 | 盼望总能凑到有差异组 | 限制补采轮数，保留真实跳过状态 | `collect_active_batch` |
| masked token 越多更新越猛 | 错把 kept token 数当分母 | 除全局 response token 数 | `policy_update`、`dppo_loss_sum` |
| 推送了权重却仍像旧策略 | prefix cache 生命周期或目标不匹配 | 校验推送数、恢复缓存后清前缀、重算 log-prob | `_wake_and_sync`、`logprob_sync_check` |
| 单卡内存撑不住两套运行时 | 生成与训练同时持有大状态 | CPU offload、sleep/wake、分块 fp32 head | `_generate`、`chunked_token_logprobs` |

## 花费

这里没有独立的正式 GPU 账单。CPU 例子不会创建云资源；单卡串行调度意在让流程满足容量约束，但没有对照测量就不能声称节省了某个百分比的费用或训练时间。作者全项目约200美元的口径见前章，不能重复分摊成每章成本。

## 教训

RL 的可信度取决于合同能否串起来：奖励属于这条轨迹，行为概率来自生成时权重，训练器看到同一串 token，优势过滤遵守既定口径，更新后下一轮真正使用新策略。每一个环节都有“能跑但训练错对象”的失败方式。

本章最值得自己改一遍的实验，是把 loss 分母换成 keep 数、把截断样本从 baseline 删除，再观察测试为何失败。这样能理解实现选择改变了什么，而不是只记住 DPPO、LoRA 或 vLLM 的名字。

## 本章出处

- `train/rl_trainer.py`：`build_train_items`、`collect_active_batch`、`policy_update`、`logprob_sync_check`、`_wake_and_sync`、`_generate`；`train/losses.py`、`train/rollout.py`、`train/agent_env.py`，提交 `ca57456`。
- `configs/rl/qwen35_2b.yaml`、`configs/rl/qwen35_4b.yaml`；`docs/apexo/08_execution_contract.md`、`docs/apexo/09_custom_training_design.md`，提交 `ca57456`。
- `tests/test_rl_step.py`、`tests/test_losses.py` 与 `docs/apexo/00_progress.md`，提交 `ca57456`。本书另写 CPU 教学测试，不将它们冒充源项目完整集成测试。
- DPPO 原论文及作者见本章开头与许可附录；Apexo 名称不替代公开算法、数据或工具归属。
