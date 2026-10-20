# 第 7 章 SFT：在 Colab 上把格式教会（10-11 至 10-13，round2）

## 动机

SFT 是本项目第一次让正式基座按成功终端轨迹更新参数。它的任务不是替代环境验证，而是让模型更熟悉思考、调用、观测与下一步行动的格式。在进入稀疏奖励的真实 RL 前，先确认数据能读、loss 有限、adapter 确实更新、checkpoint 可恢复，才有理由承担长时间运行。

这一阶段不是一次从头到尾没有中断的训练。10-11 先解决依赖问题，分别验证 2B、4B 的五步 SFT；10-12 冻结数据后重跑短链路，遇到权重链接、Drive 挂载和显存口径问题；随后用正式长度配置做五步加一步续训，最后才进行两轮 epoch 的正式 SFT。无沙箱 RL 预演穿插其中，但构造奖励不能算真实任务能力，本章仅解释它为何阻断了当时的工程流程。

主要证据为 `result/round2/` 与进度台账 §6.6–6.8、§6.21–6.27，归档提交 `ca57456`。下文区分正式 SFT 的运行代码 `f5abb46cb6a6e6c30eae5a57d1e2a713313acb0d`、早期短链路 `ffd4713` 和结项归档，不用未来文件默认值冒充当时覆盖参数。

## 配置

### 7.1 三档验证，不用五步小样本冒充正式训练

| 项目 | 早期五步 smoke | 正式长度短跑 | 正式 SFT |
| --- | --- | --- | --- |
| 目标 | 能加载、反传、保存适配器 | 给定长度配置能启动并续训 | 完成两轮 epoch 并交接 |
| 训练步数 | 5 | 5 + 从 checkpoint 续训 1 步 | 2B 702；4B 688 |
| 序列长度上限 | 8,192 | 2B 24,576；4B 20,480 | 同左 |
| 全局 batch | 2 | 短跑脚本的诊断覆盖，不等同正式吞吐 | 16 |
| 数据范围 | 诊断小子集，不是全量 | 通过长度筛选后的数据中短跑 | 2B 5,618；4B 5,509 条 |
| 能得出的结论 | 小规模更新路径存在 | 启动及续训成功，不证明打满长度 | 指定训练完成，不直接等于质量门禁通过 |

10-11 的 2B smoke 记录 loss 0.4135→0.2322，allocated 峰值记录为 11.08GB；4B 为 0.322190→0.183694、17.34 GiB，13/16 条样本保留。两者来自不同子项，不能跟第二天的另一组五步 loss 接成一条曲线。4B 当时使用代码 `1ef6c06`，还核对了 adapter 的 LoRA-B 非零及完整恢复文件。出处：`00_progress.md` §6.6、§6.8。

10-12 重跑的五步 loss 分别为：

- 2B：0.3832、0.3672、0.3299、0.3334、0.3610。
- 4B：0.3157、0.2697、0.2542、0.2380、0.2781。

它们并非每一步单调下降。验收小步训练时，应检查有限值、参数更新与产物完整性，而不是要求每个 batch 的 loss 都比前一步低。不同 batch 的有效 token 数和难度不同，单步波动本身不足以诊断故障。

### 7.2 正式运行的固定参数与覆盖项

| 项目 | 2B | 4B | 证据与边界 |
| --- | --- | --- | --- |
| 官方基座 | Qwen3.5-2B-Base | Qwen3.5-4B-Base | 官方完整 ID 及归属见许可附录 |
| model revision | b1485b2fa6dfa1287294f269f5fb618e03d52d7c | 1001bb4d826a52d1f399e183466143f4da7b741b | 两份 SFT handoff |
| SFT 数据 revision | 9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4 | 同左 | round1 与数据配置 |
| 数据 manifest SHA-256 | 879d349c827387a643e9503ccf3875a47e296e0be078d537084c074500c9437b | 同左 | 长度筛选发生在读取训练数据时，不偷偷修改原 manifest |
| dtype／注意力实现 | bfloat16／sdpa | 同左 | 对应 YAML |
| LoRA | r64、alpha64、dropout0，12 类目标 | 同左 | 第 6 章目标列表 |
| 全局 batch／epoch | 16／2 | 16／2 | 日志 world=1 |
| lr／warmup／scheduler | 2e-4／0.03／linear | 同左 | `train` 配置 |
| weight decay／grad clip | 0／1.0 | 同左 | grad_norm 日志为裁剪调用返回值，不是声称每步都被截成1 |
| max_seq_len | 24,576 | 20,480 | 正式运行配置 |
| logprob chunk | 1,024 | 同左 | fp32 head、分块计算；不是完整词表 logits 常驻 |
| 保存／保留 | 每10步／保留3份 | 同左 | save_every=10 是运行覆盖；结项 YAML 默认50不能倒写 |
| merged 导出 | false | false | Colab只交接adapter；后续导出另做 |
| 硬件 | A100-SXM4-80GB，81920 MiB | 同一顺序运行链路 | 不把原设计40GB当成这次实跑硬件 |

环境记录为 torch 2.10.0+cu128、Transformers 5.6.0、PEFT 0.18.1、vLLM 0.19.1；causal-conv1d 因 CUDA 工具链不匹配走 torch fallback。这里列 vLLM 是当时共同环境，不表示正式 SFT 的每次 forward 由 vLLM 执行。

### 7.3 为什么 5,618 条数据不是每轮 352 步

源训练器按 `len(examples) // global_batch_size` 得到每轮完整 batch 数。2B 为 5,618 // 16 = 351，2 epoch 得到 702 步；4B 为 5,509 // 16 = 344，得到 688 步。每轮不足一个完整 batch 的尾部未用于该轮更新；不能把除法向上取整后声称日志漏了一步。数据顺序由 seed 与 epoch 确定，不意味着两轮总是丢掉相同任务。

这里“全局 batch=16”是一次 optimizer step 的样本数，不是单次 forward 同时堆叠16条长序列。源实现以每次一条样本累积，再按这一步的全局有效标签 token 数归一化。不同样本长度不同时，先算各样本均值再平均，会得到另一个目标。

## 命令

### 7.4 记录中的实际入口

源 README 记录环境建立、CPU测试、两种尺寸 smoke，再到后续正式 SFT 的入口。下面保留脚本名与实际覆盖项，省略机器、账号、云目录和运行目录；这是历史命令说明，不是本书 CI 正在运行 GPU：

```{.bash .source-excerpt}
bash scripts/setup_env.sh
.venv/bin/python -m pytest tests -q
bash scripts/run_sft_smoke.sh 2b
bash scripts/run_sft_smoke.sh 4b
bash scripts/run_sft_colab.sh 2b --set output.save_every=10
bash scripts/run_sft_colab.sh 4b --set output.save_every=10
```

正式包装脚本先把 manifest 放到 VM 本地盘，模型缓存和 checkpoint 也落本地；周期性复制已完成 checkpoint 到持久存储。`run_sft_colab.sh` 使用 `--resume auto` 并强制 `output.export_merged=false`，不是每次手动从头训练，也不是在这里宣称已经得到可直接服务的 merged 权重。出处：`result/round2/scripts/run_formal_sft.py`、`scripts/run_sft_colab.sh`、两份 handoff，提交 `ca57456`。

这组历史命令要求源项目 GPU 环境和真实数据。读者本章实际运行的是下面两个离线入口：

```bash
python examples/ch07_sft_accounting.py
pytest -q tests/test_ch07_sft_accounting.py
```

### 7.5 在 CPU 上看清 token 平均与恢复状态

本书小例子不训练 Qwen，只用8个虚构标签的微型 logits 比较一次整体交叉熵与两段累积的梯度。两段都用相同的总 token 数作分母，梯度应与整体计算相同。再用一个标量 AdamW 更新，比较恢复完整 optimizer state 与只恢复权重：前者得到相同下一步，后者一般不同。

下面是 `chapter_snippet` 的原样教学代码，数值均人为构造：

```python
counts = [2, 6]
nll_sums = [4.0, 6.0]
token_mean = sum(nll_sums) / sum(counts)
sample_mean = sum(n / c for n, c in zip(nll_sums, counts)) / len(counts)
assert token_mean == 1.25
assert sample_mean == 1.5
```

对应的项目实现如下。`train/sft_trainer.py` L190–194，提交 `ca57456`，连续原样摘录；它位于真实训练循环内部，不在本书作为独立代码执行：

```{.python .source-excerpt}
            labels = torch.tensor(ex.labels[1:], device=info.device)
            mask = labels != -100
            logp = MU.sequence_logprobs(model, head_w, ids, chunk)
            loss = -(logp * mask).sum() / global_tokens
            loss.backward()
```

完整恢复还涉及 scheduler、RNG、训练步与数据顺序、配置及 manifest 校验。本书标量例子只隔离 optimizer state 的作用，不声称其简化 checkpoint 选择器实现了源项目全部哈希合同。源项目的 `.complete` 标记是候选选择条件，不能单独证明跨挂载复制已完整落盘。

![图 7-1　训练 checkpoint 与交接的不同职责：先保存可恢复状态，再核验并交接 adapter。结构示意。](images/ch07_checkpoint.svg)

## 时间线与耗时

### 7.6 同时看 UTC 和新加坡时间

SGT 为 UTC+08:00。表中“训练循环区间”以日志 `Training:` 到最后一步为端点，不包括此前安装、权重下载和此后保存／复制。毫秒来自原始日志；不是额外测量精度，也不是人工工时。

| 事件 | UTC 开始 → 结束 | SGT 开始 → 结束 | 由端点相减／状态 |
| --- | --- | --- | --- |
| 10-11 2B／4B 首轮smoke | 台账未给完整起止时间 | 仅记录日期10-11 | 不用逐步时间拼成完整会话时长 |
| 10-12 再次2B smoke | 07:06:48启动，独立结束未在README列出 | 15:06:48启动 | 五步完成；不把下一阶段启动当它的精确结束 |
| 10-12 再次4B smoke | 07:13:09启动，独立结束未在README列出 | 15:13:09启动 | 五步完成 |
| 2B 正式长度短SFT及续训 | 10-12 11:15:19 → 11:26:38 | 10-12 19:15:19 → 19:26:38 | 11分19秒；5+1步 |
| 4B 正式长度短SFT及续训 | 10-12 11:26:41 → 11:37:02 | 10-12 19:26:41 → 19:37:02 | 10分21秒；5+1步 |
| 正式包装启动 | 10-12 12:23:26；完整SFT_DONE行未保存 | 10-12 20:23:26；结束未知 | 不据日志缺行伪造结束时间 |
| 2B 正式训练循环 | 10-12 12:27:58.921 → 20:12:58.140 | 10-12 20:27:58.921 → 10-13 04:12:58.140 | 7小时44分59.219秒；702步 |
| 4B 正式训练循环 | 10-12 20:16:48.015 → 10-13 08:02:03.136 | 10-13 04:16:48.015 → 16:02:03.136 | 11小时45分15.121秒；688步 |

2B wrapper 的 `STAGE_EXIT 2b 0` 为10-12 20:13:10Z，晚于最后训练步；4B随即启动准备。两段耗时不应把这个保存／启动间隙算成“同一个optimizer step”。正式日志没有 resume 记录，说明本次完整长跑没有证据显示中途续训；短跑的第6步才是已记录的恢复测试。具备恢复功能与本轮实际恢复过，是两回事。

### 7.7 从逐步记录画图，不用两个端点制造平滑曲线

![图 7-2　正式SFT逐步loss与grad_norm：2B 702步、4B 688步，原始记录点全部保留，不平滑。](images/plots/ch07_sft_curves.png)

本图由 `data/ch07_sft_metrics.json` 生成。数据只收录两份原始 JSONL 的8个数值字段，保持全部1,390条记录及原浮点值，没有选取“好看”的片段。横轴为各自 optimizer step，不是相同墙钟时间，也不是相同任务；两种长度筛选后的样本集合不同，不能把同一横坐标当作公平逐样本对照。

`grad_norm` 是源代码裁剪函数返回的总范数；阈值1.0画作参考线，不声称曲线每个值是裁剪后的范数。`loss` 是这一步有效标签 token 的平均负对数似然。训练loss波动或下降不能直接证明终端任务成功率提高，更不能取代后续Harbor评测。

正式长度短跑也保留了不同证据：2B五步loss为0.4044、0.3225、0.3104、0.3588、0.2923，续训第6步0.2953；4B为0.2645、0.3410、0.2955、0.2758、0.3171，第6步0.2633。各步实际标注token远未填满配置上限，因此不能将这些短跑当作满长显存压力验收。

![图 7-3　正式 SFT 的 PyTorch allocated 高水位；不是整卡物理峰值，也不是最低显存需求。](images/plots/ch07_memory.png)

2B峰值13.91 GiB、4B21.64 GiB来自正式metrics的最大 `peak_mem_gb`。字段按 `torch.cuda.max_memory_allocated / 2**30` 计算；它不包含所有其他进程、驱动与推理引擎物理占用。因此不能仅凭21.64小于40就宣布4B全部流程在40GB上已验收。

## 坑与修复

下表同时标明实际事故与登记风险。R32是长期训练的恢复风险，不表示正式长跑实际发生过checkpoint损坏；后续GPU平台故障留给对应章节。

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 10-11首次依赖解析阻断，训练未开始 | vLLM0.19.1正式依赖排除Transformers5.4.0 | 固定5.6.0后重新安装与CPU验证 | `00_progress.md` §6.6 |
| causal-conv1d构建与CUDA不匹配 | cu128的torch与nvcc13主版本不一致 | auto模式编译前跳过，使用torch fallback；不声称快路径已验 | §6.6–6.8；`scripts/setup_env.sh` |
| safetensors报header too large | Drive snapshot文件只是79字节链接替身，不是真权重 | 本地重建指向真实blob的链接／下载到本地缓存 | `result/round2/README.md`；R35 |
| 首次4B预演在权重加载后停住 | 约67分钟后Drive挂载掉线，权重/adapter/日志仍依赖挂载 | 材料、基座和日志先放VM本地盘，持久存储用于复制与归档 | README；§6.21；R35 |
| 40GB卡报告42–46GiB且旧门禁失败，却没有OOM | PyTorch逻辑高水位混入vLLM sleep已释放物理页关系 | RL改为物理采样，并保留逻辑诊断；不把旧FAIL改PASS | `memory_audit_20261013.md`；R36 |
| 80GB缩小缓存仍被38GiB门禁拒绝 | 换硬件没有改变错误统计口径 | 分开硬件合同与统计来源；后来用物理门禁重新预演 | §6.21–6.24；R37 |
| reset前读数可能写到下一步 | 采样线程跨重置时刻竞争 | 世代号使跨reset读数作废；同一步采样错误仍失败 | §6.23；R36 |
| 五步正式长度配置通过，被误当满长压力证据 | 上限不是实际样本长度 | 同时报告真实label_tokens和未打满边界 | §6.25；两份短SFT metrics |
| checkpoint标记存在但远端大文件可能未落盘（风险） | FUSE异步回写与断线 | 本地完成后复制、读回哈希、完整状态恢复，失败不宣布交接 | R32；`run_formal_sft.py`；`sft_handoff.py` |
| 训练结束后VM主日志被回收 | 易失本地会话结束，最终SFT_DONE行未持久保存 | 从持久产物读回两份adapter/metrics/handoff，保留缺失事实 | §6.27；正式SFT归档 |
| 成功轨迹生产数据与all对照混用（风险） | 数据选择合同不同 | all只作独立对照，交接名单外数据必须显式核验 | R33；数据计划 |

物理显存采样本身也不是绝对真值：50ms采样可能漏掉瞬时尖峰，仍要结合无OOM、余量和正式长度/并发压力。round2的“预演通过”只代表无沙箱引擎链路，不能算真实任务训练或Harbor质量验收。源码归档中的后续40GB策略不要倒写成当时已经验证；第8章再解释算力路线变化。

## 花费

这一阶段使用过Colab A100 40GB与80GB，正式SFT跑在80GB上。源记录没有足以核对每段训练美元费用的账单，不能把两个训练循环区间乘一个猜测的单价；也不能把某一时刻显示的费率当作全阶段免费凭据。

作者10-19补充的约200美元是全项目Google Cloud口径，Colab计算单元不在其中。本章不拆分金额，不将该总额分配给SFT或预演；Spot／普通实例也不从Colab机器名推断。出处：`docs/apexo/11_timeline.md`“作者补充”，提交 `ca57456`。

## 教训

1. **先确认更新，再延长训练。** 五步smoke、正式长度pilot、恢复测试和完整epoch各自回答不同问题。不能把其中一个PASS扩大成全部流程PASS。
2. **记录覆盖项，不只记录配置文件名。** save_every从默认50覆盖到10、export_merged强制false，会改变产物与恢复行为；必须与handoff一起保存。
3. **恢复需要状态，不只是权重。** adapter、optimizer、scheduler、RNG、步数、数据顺序及配置哈希共同决定接下来算什么。
4. **指标名必须带统计来源。** SFT allocated高水位、旧RL逻辑峰值、新RL物理采样峰值不能直接相加或横比。
5. **交接与质量分开。** 两份adapter及metrics哈希一致支持产物完整性；正式SFT完成支持训练过程完成；真实任务能力仍要等待同口径评测。没有保存的最终日志行，不用推测补齐。

## 本章出处

- `docs/apexo/00_progress.md` §6.6–6.8、§6.21–6.27；`06_risks_and_decisions.md` R32、R33、R35–R37；`11_timeline.md`。提交 `ca57456`。
- `result/round2/README.md`、`logs/formal-sft.log`、`results/formal_sft/colab_sft_2b.log`、`colab_sft_4b.log`：启动、实际循环时间、结果与未保存部分。提交 `ca57456`。
- 同目录 `metrics_2b.jsonl`、`metrics_4b.jsonl`、`sft_handoff_2b.json`、`sft_handoff_4b.json`；`results/sft_len_2b_metrics.jsonl`、`sft_len_4b_metrics.jsonl`、`results/memory_audit_20261013.md`。归档 `ca57456`，正式运行代码 `f5abb46`。
- `scripts/run_sft_colab.sh`、`scripts/run_sft_smoke.sh`、`result/round2/scripts/run_formal_sft.py`；`configs/sft/qwen35_2b_lora.yaml`、`qwen35_4b_lora.yaml`；`train/sft_trainer.py` L190–194、`train/dataset.py`。提交 `ca57456`。
- `data/ch07_sft_metrics.json` 为原始数值的脱敏摘录；`examples/ch07_sft_accounting.py` 为本书原创CPU实验，不是新一轮模型训练。
