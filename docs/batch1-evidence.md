# 批次 1：事实与证据索引

范围：第 0–1 章。署名 Hawk Wu；教学代码及示意图为本批新写，不是项目测量。项目事实以下均指源仓库提交 `ca57456`，只给纯文本路径，不提供私有链接。未复制原始日志或云资源标识。

| 正文事实 / 数字 | 原始来源 | 使用限制 |
| --- | --- | --- |
| 2026-10-10 至 10-19；round3 未执行；round4–7 同日 | `docs/apexo/11_timeline.md` | 日历跨度不等于计算时长 |
| 官方 2B / 4B Base；完整闭环仅 2B | `result/round10/conclusion.md` 最终总结 §1；`configs/sft/qwen35_2b_lora.yaml`、`configs/sft/qwen35_4b_lora.yaml` | 不补 Base 选型动机 |
| 89 任务 × 5 次 = 445 trials | `result/round10/conclusion.md` 最终总结 §2.1 | 每阶段相同评测规模 |
| Base pass@1/3/5 均 0.00%，有效调用率 88.96% | 同上，Base 列 | 按记录显示精度照抄 |
| SFT pass@1/3/5 为 1.12% / 3.03% / 4.49%，有效调用率 96.58% | 同上，SFT 列 | 不替代 JSON 原始精度 |
| RL pass@1/3/5 为 0.90% / 2.36% / 3.37%，有效调用率 96.24% | 同上，RL 列 | 不宣称 RL 提升 |
| RL 8 步，5 步非零奖励更新 | 同上 §2.2；`result/round10/rl-2b/continue-final/metrics.jsonl` | 步号不是有效更新数；本批未逐行展开 metrics |
| 10-19 修改门禁，两次重新 PASSED，之前 FAILED | `docs/apexo/11_timeline.md`；`result/round10/harbor-sft-2b/sft-gate-apexo.json`、`result/round10/harbor-rl-2b/rl-gate-apexo.json` 的 `gate` | 不是训练前预注册；未改原始数值 |
| RL 判定容许 pass@1 差值 ≥ −0.01 | 上述 RL JSON，`gate.thresholds.rl_min_delta` | 点估计门槛不是统计不劣效证明 |
| Attempt 1 记录不完整且被覆盖 | `docs/apexo/11_timeline.md`“记录中的缺口” | 只说明缺口，不补造运行细节 |
| torch 2.10.0、transformers 5.6.0、peft 0.18.1、vllm 0.19.1 | `requirements.txt` | 属于源项目 GPU 环境，不等同本批 CPU 依赖 |
| bf16、fp32 头、r=64、alpha=64、dropout=0、全局 batch 16 | `configs/sft/qwen35_2b_lora.yaml` | 配置值不是测量值 |
| LoRA 的 12 个目标模块名 | 上述配置；`train/model_utils.py::DEFAULT_LORA_TARGETS` | 不能泛化为其他模型的模块名 |
| bs=1，无 padding/packing，按全局监督 token 数归一化，统一裁剪后 step | `train/sft_trainer.py::train` | 示例只演示单设备；不能额外重复平均 |
| 冻结头 fp32 副本、分块 log-probability、移位 | `train/model_utils.py::frozen_head_weight` / `sequence_logprobs`；`train/losses.py::chunked_token_logprobs` | 不提供新显存测量 |
| adapter、优化器、scheduler、随机状态与完成标记 | `train/sft_trainer.py::save_checkpoint` 和恢复分支 | CPU 示例不是云端原子 checkpoint 验证 |
| 监督生成前缀之后至结束标记；有效 bash 轮次；分段 tokenize；超长丢弃 | `train/dataset.py` | 教学 mask 不是完整的数据清洗器 |
| Auto 加载、固定 revision、安装模板、关闭 cache、non-reentrant checkpoint | `train/model_utils.py::load_policy` / `load_tokenizer` | 玩具 GPT2 不验证正式 Qwen 模型 |

| 作者补充总花费约 200 美元、全部 GCP、Colab 计算单元不在内；4B 因成本停止 | `docs/apexo/11_timeline.md` 作者补充第 1、2 条 | 作者 2026-10-19 补充，无逐轮账单，不能拆分 |

## 教学数字与公开技术来源

张量形状、字节数、标量 SGD 的 0 → 0.4、2/5 token 分母例子、低秩矩阵维度、脚本随机种子、微型词表与模型参数均为人为构造，非项目记录。可以从随书代码直接复算。数学讲解是独立解释，未摘抄第三方代码。

- PyTorch 2.10 自动微分：https://docs.pytorch.org/docs/2.10/notes/autograd.html
- PyTorch 2.10 checkpoint：https://docs.pytorch.org/docs/2.10/checkpoint.html
- PyTorch 2.10 CrossEntropyLoss：https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html
- Hugging Face chat templates：https://huggingface.co/docs/transformers/chat_templating
- Hu 等，LoRA: Low-Rank Adaptation of Large Language Models：https://arxiv.org/abs/2106.09685

Apexo 用作项目流程名，不据此删除公开算法归属。本批未复制或重发外部数据集、harness、模型权重；附录 E 按既定批次后续交付，必须依源 `THIRD_PARTY.md` 保留相应署名与许可证，不能以重命名流程替代这些义务。

## 作者问题

本批没有新增问题。已采用最新 `docs/outline.md` 的范围与作者补充状态；未写成本细分、未记录的选型动机、未知运行耗时。本批不扩写第 2–15 章。大纲 v3 为后续第 6、13 章新增 9 个作者问题，留待相应批次整理，不伪称全书问题清零。

## 版本核对

发布前发现书仓库大纲 v2、v3 已合并，因此与源提交 `5eec66a` 比较后更新为 `ca57456`。本批所引训练代码、SFT 配置、依赖、THIRD_PARTY 与两个重判 JSON 未变；时间线新增作者回答，结项报告仅补充早期资源状态的后来变更，最终评测表未变。本批不引用资源标识或新增动机。

## 审核修订：源码摘录

新增静态 `.source-excerpt` 阅读片段，均来自 `ca57456`：`train/sft_trainer.py` L183–200；`train/losses.py` L112–130；`train/model_utils.py` L95–122；`train/dataset.py` L3–16（数据集 ID 作明示替换，原名保留在许可附录）；`configs/sft/qwen35_2b_lora.yaml` L7–19、L27–36。未包含资源标识、凭据或首行外部数据集注释。静态片段不执行，教学代码仍单独逐字测试。

## 实测图与示意图

`assets/images/ch00_metrics.svg` 由 `scripts/draw_ch01.py` 根据 `scripts/data/batch1_metrics.json` 生成；该 JSON 是 `result/round10/conclusion.md` 最终总结 §2.1、提交 `ca57456` 的脱敏展示值摘录，保留原百分数字符串。其余四图是中文教学示意，不是测量。
