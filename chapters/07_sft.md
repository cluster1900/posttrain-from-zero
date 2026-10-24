# 第 7 章 SFT：在 Colab 上把格式教会（10-11 至 10-13，round2）

## 动机

SFT 是本项目第一次让正式基座按成功终端轨迹更新参数，目标是让模型熟悉思考、调用、观测与下一步行动的格式。在进入稀疏奖励的真实 RL 前，先确认数据能读、loss 有限、adapter 确实更新、checkpoint 可恢复，再承担长时间运行。

这一阶段不是一次从头到尾没有中断的训练。10-11 先解决依赖问题，分别验证 2B、4B 的五步 SFT；10-12 冻结数据后重跑短链路，遇到权重链接、Drive 挂载和显存口径问题；随后用正式长度配置做五步加一步续训，最后才进行两轮 epoch 的正式 SFT。无沙箱 RL 预演穿插其中，本章只讲它怎样阻断了当时的工程流程。

主要证据为 `result/round2/` 与进度台账 §6.6–6.8、§6.21–6.27，归档提交 `ca57456`。下文区分正式 SFT 的运行代码 `f5abb46cb6a6e6c30eae5a57d1e2a713313acb0d`、早期短链路 `ffd4713` 和结项归档；参数一律取当时的覆盖值，而不是结项时文件里的默认值。

## 配置

### 7.1 三档验证

| 项目 | 早期五步 smoke | 正式长度短跑 | 正式 SFT |
| --- | --- | --- | --- |
| 目标 | 能加载、反传、保存适配器 | 给定长度配置能启动并续训 | 完成两轮 epoch 并交接 |
| 训练步数 | 5 | 5 + 从 checkpoint 续训 1 步 | 2B 702；4B 688 |
| 序列长度上限 | 8,192 | 2B 24,576；4B 20,480 | 同左 |
| 全局 batch | 2 | 短跑脚本的诊断覆盖 | 16 |
| 数据范围 | 诊断小子集，不是全量 | 通过长度筛选后的数据中短跑 | 2B 5,618；4B 5,509 条 |
| 能得出的结论 | 小规模更新路径存在 | 该长度配置能启动并续训 | 两轮 epoch 完成并交接 |

10-11 的 2B smoke 记录 loss 0.4135→0.2322，allocated 峰值记录为 11.08GB；4B 为 0.322190→0.183694、17.34 GiB，13/16 条样本保留。两者来自不同子项，和第二天的另一组五步 loss 分开看。4B 当时使用代码 `1ef6c06`，还核对了 adapter 的 LoRA-B 非零及完整恢复文件。出处：`00_progress.md` §6.6、§6.8。

10-12 重跑的五步 loss 分别为：

- 2B：0.3832、0.3672、0.3299、0.3334、0.3610。
- 4B：0.3157、0.2697、0.2542、0.2380、0.2781。

它们并非每一步单调下降。验收小步训练看的是有限值、参数更新与产物完整性；不同 batch 的有效 token 数和难度不同，单步 loss 有升有降是正常的。

### 7.2 正式运行的固定参数与覆盖项

| 项目 | 2B | 4B | 证据与边界 |
| --- | --- | --- | --- |
| 官方基座 | Qwen3.5-2B-Base | Qwen3.5-4B-Base | 官方完整 ID 及归属见许可附录 |
| model revision | b1485b2fa6dfa1287294f269f5fb618e03d52d7c | 1001bb4d826a52d1f399e183466143f4da7b741b | 两份 SFT handoff |
| SFT 数据 revision | 9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4 | 同左 | round1 与数据配置 |
| 数据 manifest SHA-256 | 879d349c827387a643e9503ccf3875a47e296e0be078d537084c074500c9437b | 同左 | 长度筛选发生在读取训练数据时，原 manifest 保持不变 |
| dtype／注意力实现 | bfloat16／sdpa | 同左 | 对应 YAML |
| LoRA | r64、alpha64、dropout0，12 类目标 | 同左 | 第 6 章目标列表 |
| 全局 batch／epoch | 16／2 | 16／2 | 日志 world=1 |
| lr／warmup／scheduler | 2e-4／0.03／linear | 同左 | `train` 配置 |
| weight decay／grad clip | 0／1.0 | 同左 | grad_norm 日志是裁剪调用返回的裁剪前总范数 |
| max_seq_len | 24,576 | 20,480 | 正式运行配置 |
| logprob chunk | 1,024 | 同左 | fp32 head、按 1,024 个位置分块计算 logits |
| 保存／保留 | 每10步／保留3份 | 同左 | save_every=10 是运行覆盖；结项 YAML 默认是 50 |
| merged 导出 | false | false | Colab只交接adapter；后续导出另做 |
| 硬件 | A100-SXM4-80GB，81920 MiB | 同一顺序运行链路 | 原设计目标是 40GB，这次实跑在 80GB |

环境记录为 torch 2.10.0+cu128、Transformers 5.6.0、PEFT 0.18.1、vLLM 0.19.1；causal-conv1d 因 CUDA 工具链不匹配走 torch fallback。vLLM 是同一环境里给 RL 预演用的，正式 SFT 的 forward 由 PyTorch 执行。

### 7.3 为什么 5,618 条数据不是每轮 352 步

源训练器按 `len(examples) // global_batch_size` 得到每轮完整 batch 数。2B 为 5,618 // 16 = 351，2 epoch 得到 702 步；4B 为 5,509 // 16 = 344，得到 688 步。每轮不足一个完整 batch 的尾部（2B 2 条，4B 5 条）不参与该轮更新，所以日志没有漏步。数据顺序由 seed 与 epoch 确定，两轮丢掉的尾部样本一般不同。

这里“全局 batch=16”是一次 optimizer step 的样本数；源实现每次 forward 只放一条样本，累积 16 次后按这一步的全局有效标签 token 数归一化。不同样本长度不同时，先算各样本均值再平均，会得到另一个目标。7.8 节会看到这个设计在耗时和显存上的直接后果。

## 命令

### 7.4 记录中的实际入口

源 README 记录环境建立、CPU测试、两种尺寸 smoke，再到后续正式 SFT 的入口。下面保留脚本名与实际覆盖项，省略机器、账号、云目录和运行目录：

```{.bash .source-excerpt}
bash scripts/setup_env.sh
.venv/bin/python -m pytest tests -q
bash scripts/run_sft_smoke.sh 2b
bash scripts/run_sft_smoke.sh 4b
bash scripts/run_sft_colab.sh 2b --set output.save_every=10
bash scripts/run_sft_colab.sh 4b --set output.save_every=10
```

正式包装脚本先把 manifest 放到 VM 本地盘，模型缓存和 checkpoint 也落本地；周期性复制已完成 checkpoint 到持久存储。`run_sft_colab.sh` 使用 `--resume auto` 并强制 `output.export_merged=false`：中断后自动从最近的完整 checkpoint 续训，这一阶段只交接 adapter，merged 权重留到后续导出。出处：`result/round2/scripts/run_formal_sft.py`、`scripts/run_sft_colab.sh`、两份 handoff，提交 `ca57456`。

这组命令需要源项目 GPU 环境和真实数据。读者在本章运行的是下面两个离线入口：

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

完整恢复还涉及 scheduler、RNG、训练步与数据顺序、配置及 manifest 校验；本书标量例子只隔离 optimizer state 的作用。源项目的 `.complete` 标记是候选 checkpoint 的选择条件，跨挂载复制是否完整落盘另靠读回哈希确认（见坑表）。

![图 7-1　训练 checkpoint 与交接的不同职责：先保存可恢复状态，再核验并交接 adapter。结构示意。](images/ch07_checkpoint.svg)

## 时间线与耗时

### 7.6 同时看 UTC 和新加坡时间

SGT 为 UTC+08:00。表中“训练循环区间”以日志 `Training:` 到最后一步为端点，不含此前安装、权重下载和此后保存／复制。毫秒来自原始日志。

| 事件 | UTC 开始 → 结束 | SGT 开始 → 结束 | 由端点相减／状态 |
| --- | --- | --- | --- |
| 10-11 2B／4B 首轮smoke | 台账未给完整起止时间 | 仅记录日期10-11 | 起止未知 |
| 10-12 再次2B smoke | 07:06:48启动，独立结束未在README列出 | 15:06:48启动 | 五步完成；结束时间未记录 |
| 10-12 再次4B smoke | 07:13:09启动，独立结束未在README列出 | 15:13:09启动 | 五步完成 |
| 2B 正式长度短SFT及续训 | 10-12 11:15:19 → 11:26:38 | 10-12 19:15:19 → 19:26:38 | 11分19秒；5+1步 |
| 4B 正式长度短SFT及续训 | 10-12 11:26:41 → 11:37:02 | 10-12 19:26:41 → 19:37:02 | 10分21秒；5+1步 |
| 正式包装启动 | 10-12 12:23:26；完整SFT_DONE行未保存 | 10-12 20:23:26；结束未知 | 结束行缺失 |
| 2B 正式训练循环 | 10-12 12:27:58.921 → 20:12:58.140 | 10-12 20:27:58.921 → 10-13 04:12:58.140 | 7小时44分59.219秒；702步 |
| 4B 正式训练循环 | 10-12 20:16:48.015 → 10-13 08:02:03.136 | 10-13 04:16:48.015 → 16:02:03.136 | 11小时45分15.121秒；688步 |

2B wrapper 的 `STAGE_EXIT 2b 0` 为10-12 20:13:10Z，晚于最后训练步；4B随即启动准备。两段耗时都不含这个保存／启动间隙。正式日志没有 resume 记录，完整长跑中途没有续训；已记录的恢复测试是短跑的第 6 步。

### 7.7 逐步记录

![图 7-2　正式SFT逐步loss与grad_norm：2B 702步、4B 688步，原始记录点全部保留，不平滑。](images/plots/ch07_sft_curves.png)

本图由 `data/ch07_sft_metrics.json` 生成，收录两份原始 JSONL 的 8 个数值字段，全部 1,390 条记录、原浮点值。横轴是各自的 optimizer step；两种长度筛选后的样本集合不同，同一横坐标上的 2B 与 4B 并非同一批样本。

`grad_norm` 是源代码裁剪函数返回的裁剪前总范数，阈值 1.0 画作参考线。`loss` 是这一步有效标签 token 的平均负对数似然；终端任务成功率要看后续 Harbor 评测。

正式长度短跑也保留了不同证据：2B五步loss为0.4044、0.3225、0.3104、0.3588、0.2923，续训第6步0.2953；4B为0.2645、0.3410、0.2955、0.2758、0.3171，第6步0.2633。各步标注 token 只有 3,283–11,846，远未填满配置上限，所以这些短跑没有测到满长样本的显存。

![图 7-3　正式 SFT 的 PyTorch allocated 高水位（torch.cuda.max_memory_allocated）。](images/plots/ch07_memory.png)

2B 峰值 13.91 GiB、4B 21.64 GiB 来自正式 metrics 的最大 `peak_mem_gb`，按 `torch.cuda.max_memory_allocated / 2**30` 计算，只统计本进程 PyTorch 分配器，不含其他进程、驱动和推理引擎的物理占用。第 6 章把这两个峰值拆成了权重、fp32 输出头、LoRA 状态和激活四部分。

### 7.8 每步耗时由标注 token 数决定，显存不是

![图 7-4　正式 SFT 每步耗时与该步标注 token 数（第 1 步预热除外）：2B 701 步、4B 687 步全部画出，直线为最小二乘拟合。数据：data/ch07_sft_metrics.json。](images/plots/ch07_time_vs_tokens.svg)

去掉含编译和缓存预热的第 1 步（2B 315.03 秒、4B 238.02 秒），对其余各步做最小二乘拟合：

| 拟合项 | 2B（701 步） | 4B（687 步） |
| --- | --- | --- |
| 斜率（秒／千标注 token） | 0.526 | 0.848 |
| 截距（秒／步） | 11.84 | 18.86 |
| 耗时与标注 token 的相关系数 r | 0.852 | 0.865 |
| 每秒标注 token 中位数 | 1,326 | 811 |
| 显存峰值与标注 token 的相关系数 | 0.018 | 0.014 |
| 全部步的 `step_time_s` 之和 | 27,741.48 秒 | 42,098.11 秒 |

每步耗时几乎线性地跟着这一步的标注 token 数走：同样 1,000 个标注 token，4B 比 2B 多花约 1.6 倍时间（0.848 / 0.526）。截距是与 token 数无关的固定开销，包括 16 次 forward/backward 的启动和 optimizer step。把 `step_time_s` 全部加起来，2B 是 7 小时 42 分，4B 是 11 小时 41 分，与 7.6 节日志端点相减的 7 小时 44 分 59 秒、11 小时 45 分 15 秒只差 2–4 分钟，差额是保存 checkpoint 等步外时间。所以按“总标注 token × 斜率 + 步数 × 截距”可以事先估算一次 SFT 要跑多久。

显存则完全不同：峰值与这一步的标注 token 总数几乎不相关（r≈0.01–0.02）。原因在 7.3 节：每次 forward 只放一条样本，峰值由这一步里最长的那一条决定，跟 16 条加起来有多少 token 无关。要估显存上限，看的是 `max_seq_len` 和数据里最长样本，不是全局 batch。



## 边界说明

本章的数字都是 round2 一次实际运行的记录：loss 与 grad_norm 说明训练过程正常，不说明终端任务能力；allocated 峰值是本进程分配器口径，不是整卡物理占用或最低硬件需求，4B 在 40GB 卡上的完整流程要以第 8 章的物理显存门禁为准；无沙箱 RL 预演用的是构造奖励，只验证引擎链路。

## 坑与修复

下表同时列出实际事故与登记风险（标“风险”的行）。R32 是长期训练的恢复风险，正式长跑没有发生 checkpoint 损坏；后续 GPU 平台故障见第 8 章。

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 10-11首次依赖解析阻断，训练未开始 | vLLM0.19.1正式依赖排除Transformers5.4.0 | 固定5.6.0后重新安装与CPU验证 | `00_progress.md` §6.6 |
| causal-conv1d构建与CUDA不匹配 | cu128的torch与nvcc13主版本不一致 | auto模式编译前跳过，使用torch fallback；快路径未验证 | §6.6–6.8；`scripts/setup_env.sh` |
| safetensors报header too large | Drive snapshot文件只是79字节链接替身，不是真权重 | 本地重建指向真实blob的链接／下载到本地缓存 | `result/round2/README.md`；R35 |
| 首次4B预演在权重加载后停住 | 约67分钟后Drive挂载掉线，权重/adapter/日志仍依赖挂载 | 材料、基座和日志先放VM本地盘，持久存储用于复制与归档 | README；§6.21；R35 |
| 40GB卡报告42–46GiB且旧门禁失败，却没有OOM | PyTorch逻辑高水位混入vLLM sleep已释放物理页关系 | RL改为物理采样，并保留逻辑诊断；旧记录保持 FAIL | `memory_audit_20261013.md`；R36 |
| 80GB缩小缓存仍被38GiB门禁拒绝 | 换硬件没有改变错误统计口径 | 分开硬件合同与统计来源；后来用物理门禁重新预演 | §6.21–6.24；R37 |
| reset前读数可能写到下一步 | 采样线程跨重置时刻竞争 | 世代号使跨reset读数作废；同一步采样错误仍失败 | §6.23；R36 |
| 五步正式长度配置通过，被误当满长压力证据 | 上限不是实际样本长度 | 同时报告真实label_tokens和未打满边界 | §6.25；两份短SFT metrics |
| checkpoint标记存在但远端大文件可能未落盘（风险） | FUSE异步回写与断线 | 本地完成后复制、读回哈希、完整状态恢复，失败不宣布交接 | R32；`run_formal_sft.py`；`sft_handoff.py` |
| 训练结束后VM主日志被回收 | 易失本地会话结束，最终SFT_DONE行未持久保存 | 从持久产物读回两份adapter/metrics/handoff，保留缺失事实 | §6.27；正式SFT归档 |
| 成功轨迹生产数据与all对照混用（风险） | 数据选择合同不同 | all只作独立对照，交接名单外数据必须显式核验 | R33；数据计划 |

物理显存每 50ms 采样一次，可能漏掉瞬时尖峰，所以还要结合是否 OOM、余量和正式长度／并发压力一起看。第 8 章解释后来的算力路线变化。

## 花费

这一阶段使用过 Colab A100 40GB 与 80GB，正式 SFT 跑在 80GB 上。可核对的资源用量是 GPU 时长：两段正式训练循环共 19 小时 30 分 14 秒（7.6 节），两段正式长度短跑共 21 分 40 秒。源记录没有逐段的美元账单，本章金额记为 null。

作者 10-19 补充的约 200 美元是全项目 Google Cloud 口径，Colab 计算单元不在其中，所以也不分摊到本章。出处：`docs/apexo/11_timeline.md`“作者补充”，提交 `ca57456`。

## 教训

1. **先确认更新，再延长训练。** 五步smoke、正式长度pilot、恢复测试和完整epoch各自回答不同问题，每一档单独验收。
2. **记录覆盖项，不只记录配置文件名。** save_every从默认50覆盖到10、export_merged强制false，会改变产物与恢复行为；必须与handoff一起保存。
3. **恢复需要状态，不只是权重。** adapter、optimizer、scheduler、RNG、步数、数据顺序及配置哈希共同决定接下来算什么。
4. **指标名必须带统计来源。** SFT allocated高水位、旧RL逻辑峰值、新RL物理采样峰值不能直接相加或横比。
5. **交接与质量分开。** 两份adapter及metrics哈希一致说明产物完整；正式SFT完成说明训练过程完成；真实任务能力要看同口径评测。没有保存的最终日志行就记为缺失。
6. **用逐步记录估算下一次。** 每步耗时≈截距＋斜率×标注 token（2B 0.526、4B 0.848 秒／千 token），显存看最长单条样本，这两条都能在开跑前算出来。

## 本章出处

- `docs/apexo/00_progress.md` §6.6–6.8、§6.21–6.27；`06_risks_and_decisions.md` R32、R33、R35–R37；`11_timeline.md`。提交 `ca57456`。
- `result/round2/README.md`、`logs/formal-sft.log`、`results/formal_sft/colab_sft_2b.log`、`colab_sft_4b.log`：启动、实际循环时间、结果与未保存部分。提交 `ca57456`。
- 同目录 `metrics_2b.jsonl`、`metrics_4b.jsonl`、`sft_handoff_2b.json`、`sft_handoff_4b.json`；`results/sft_len_2b_metrics.jsonl`、`sft_len_4b_metrics.jsonl`、`results/memory_audit_20261013.md`。归档 `ca57456`，正式运行代码 `f5abb46`。
- `scripts/run_sft_colab.sh`、`scripts/run_sft_smoke.sh`、`result/round2/scripts/run_formal_sft.py`；`configs/sft/qwen35_2b_lora.yaml`、`qwen35_4b_lora.yaml`；`train/sft_trainer.py` L190–194、`train/dataset.py`。提交 `ca57456`。
- `data/ch07_sft_metrics.json` 为原始数值的脱敏摘录；`scripts/figures/ch05_07_real_data.py` 从它计算 7.8 节的拟合；`examples/ch07_sft_accounting.py` 为本书原创 CPU 实验。
