# 《后训练从 0 到 1》详细大纲（v4，待写作）

作者：Hawk Wu

本书根据作者的真实项目 `qwen-posttrain`（以下简称"源仓库"，当前为私有仓库）整理：在官方 `Qwen/Qwen3.5-2B-Base` / `Qwen3.5-4B-Base` 上，用作者的 **Apexo 后训练流程**（作者在阅读多方公开论文与开源实现的基础上自行设计和修改）做终端智能体（Terminal Agent）后训练，从 2026-10-10 第一次提交到 2026-10-19 结项。全书按"先打基础，再按时间讲真实过程"组织：每一步写清楚**做了什么、为什么这么做、遇到了什么坑、怎么解决、花了多久、之后怎么调整**。

## 写作规则

1. **每个事实都要能追溯**到源仓库的文件、日志、指标或提交（写明路径或 commit）。数字（loss、显存、耗时、pass@k）照抄原始记录，不四舍五入成别的意思。
2. **记录里没有的不编**。原因、心路、费用等如果找不到，就在章末"待作者确认"里列问题，不要替作者补动机。唯一的例外是下面「作者问题 Q1–Q10」：作者 2026-10-19 指定由写作者（dot）分析回答，但答案必须分成三类并在正文里**显式标注**：
   - 【记录事实】：源仓库文件、日志或提交里写着的事，注明路径或提交号；
   - 【读者运行所得】：本书 `examples/` 脚本现场算出的数字（例如参数量），注明脚本名；
   - 【作者事后分析】：记录里没有写当时为什么这样做，由写作者结合常见做法推断出的理由。必须写成"事后分析"的口吻（例如"一个合理的解释是……"），不能写成"作者当时决定……"这类历史事实。
3. **如实写结果**，包括不好看的数字：RL 后 pass@1（0.90%）比 SFT（1.12%）低，必须写。门禁按 Apexo 阈值描述（见第 13 章），所有实测数字照抄原始报告，一个都不改。不要写成"阈值在训练前就定好了"，记录里也没有这个说法。
4. **方法命名**：全书统一叫"Apexo 后训练流程""Apexo RL"等。不要把方法归属给某篇外部论文或某个外部项目。公开数据集和评测 harness 是固定版本的外部输入，在附录 E 按许可证署名。
5. 公开书稿里**不出现**云项目编号、bucket 名、Drive 文件夹 ID、服务器主机名、邮箱、密钥或任何凭据；用"项目 bucket""训练服务器"等代称。
6. 风格：中文、面向能写 Python 但没做过后训练的读者；概念先用直觉解释，再给最小代码；源仓库代码只摘关键片段并注明出处。
7. 时间：原始日志多为 UTC，正文统一写成"日期 + 时长"，需要具体时刻时注明时区。
8. **引用源仓库**：源仓库保持私有。书里以纯文本写出路径和提交号（例如 `docs/apexo/11_timeline.md`、提交 `ca57456`），不放读者打不开的链接，也不依赖读者去看源仓库。

## 章节写法（必须照做）

这本书要**写得很细**。读者只看书、不看源仓库，也要能明白每一步做了什么、为什么、花了多久、出了什么问题。

**训练阶段章节**（第 3、4、7、8、9、10、11、12 章）每章都必须有下面这些二级小节，标题照抄，顺序不限：

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

**专题章节**（第 6、13 章）必须有各自规定的小节，见下面两章的「本章必备小节」。

其他章节（第 1、2、5、14、15 章和附录）至少要有 `## 本章出处`。前言除外。

**代码示例**：
- 放在 `examples/chNN_<主题>.py`，必须在 CPU 上几秒内跑完，不需要 GPU、网络或凭据。
- 每个示例都要有对应测试 `tests/test_chNN_<主题>.py`（pytest），并断言关键数值或形状。
- 正文里引用的示例代码必须和 `examples/` 里的文件一致。
- 依赖固定在 `requirements-examples.txt`。

## 配图要求（必须照做）

作者 2026-10-19 要求：**图要多，不要大段干巴巴的文字**。

- **数量**：第 1–15 章每章至少 **2 张图**（mermaid 图或图片都算；表格另算，不计入图数）。专题章节（第 6、13 章）和训练阶段章节建议 4 张以上。附录不强制。
- **类型**（按章节选用）：
  - mermaid 流程图、时序图、架构图（流程总览、一次 RL step 的时序、训练器 / vLLM / 沙箱之间的调用关系、算力方案变化）；
  - 全项目训练时间线（mermaid `gantt` 或 `timeline`，时间来自 `docs/apexo/11_timeline.md`）；
  - **真实指标曲线**：SFT loss、grad_norm、RL reward_mean、KL、entropy、步时、显存等，用 matplotlib 从源仓库的真实日志 / 指标文件画出；
  - Qwen3.5-2B 模块树，**高亮 LoRA 目标模块**和冻结部分；
  - 门禁判定流程图（每个门禁的条件、原阈值和修订阈值、PASSED / FAILED 分支）；
  - 对比表（Base / SFT / RL 指标、原阈值与修订阈值、40GB 与 80GB 方案等）。
- **mermaid 写法**：```` ```{.mermaid #fig-ch06-tree caption="图 6-1　……"} ````，构建时预渲染成 PNG（`build.py`），网页、EPUB、PDF 显示同一张图。图注格式"图 章-序号　说明"。
- **数据图**：
  - 脚本放 `scripts/figures/chNN_<名字>.py`，用 `--out <目录>` 参数输出 `chNN_<名字>.png`；生成好的图提交到 `assets/images/plots/`，正文用 `![图 7-1　……](images/plots/ch07_sft_loss.png)` 引用。
  - 脚本开头必须有 `DATA_SOURCE = "源仓库路径（提交 ca57456）"`，正文图注或下一段也写明数据来源。
  - 画图用的原始数据从源仓库复制到本书 `data/`（只复制需要的字段；不得含项目编号、bucket、主机名、邮箱、密钥），`data/README.md` 写明每个文件来自源仓库哪个路径和提交。
  - **不许编数**：曲线上每个点都必须来自原始文件；没有数据就不画，不插值、不平滑后冒充原值（平滑要另画一条并在图注说明窗口大小）。
  - 必须能在 CPU 上几秒内重跑；CI 会逐个运行 `scripts/figures/*.py`。中文用 Noto Sans CJK SC 字体。
- **构建检查**：`check_book.py` A6 检查每章图数和图片引用存在，A7 检查 `assets/images/plots/` 下每张图都有同名脚本且脚本声明了 `DATA_SOURCE`。

## 验收标准（审核时逐条检查，任何一条不满足就退回）

| # | 标准 | 怎么检查 |
| --- | --- | --- |
| 1 | **每个数字都能追溯**：每个数字、时间、配置值都能在源仓库的文件或提交里找到原文；正文或「本章出处」写明路径或提交号 | 审核时每章至少抽查 10 个数字（不足 10 个则全查），只要有一个找不到原文就退回 |
| 2 | **不编造**：动机、费用、心路只写记录里有的或作者已确认的；门禁按 Apexo 阈值写，但不写成训练前预设；实测数字照抄 | 人工审核 |
| 3 | **章节结构完整**：训练阶段章节 8 个小节齐全；第 6、13 章的必备小节齐全，第 13 章「信号总表」用规定表头；「坑与修复」用规定表头，每个坑都有证据 | `python3 scripts/check_book.py`（A3、A4）+ 人工核对 R1–R49 覆盖 |
| 4 | **代码示例能在 CPU 上跑且有测试** | CI 运行 `pytest`；`check_book.py` A5 检查每个示例都有测试 |
| 5 | **构建、Pages、EPUB、PDF 都通过** | 本地 `python3 build.py` 成功；PR 的 CI `build` job 绿（含 `check_book.py`、示例测试、画图脚本重跑、`check_outputs.py`、`epubcheck`）；合并后 Pages 部署成功 |
| 6 | **没有旧方法名**：全书只用"Apexo"命名；外部数据集和 harness 的 ID 只出现在致谢与许可附录（文件名含 `credits`、`appendix_e` 或 `99_`） | `check_book.py` A1 |
| 7 | **没有秘密和项目标识**：没有云项目编号、bucket、服务账户、镜像仓库地址、邮箱、Drive 文件夹 ID、主机名、私钥或 API key | `check_book.py` A2 + 人工审核 |
| 8 | **写得够细**：读者不看源仓库也能复述这一步的做法、原因、耗时和每个坑 | 人工审核；明显只列要点、不讲过程的章节退回 |
| 9 | **成败判定讲透**（第 13 章）：训练和评测里用到的每个信号都进「信号总表」，写清公式（与源码函数逐字对应）、样本量（89 任务 × 5 次等）、数据来源、原阈值和修订阈值、为什么这样算、证据路径；两套阈值下的结论都写 | `check_book.py` A3 检查表头和小节；人工逐行对照 `evaluation/run_eval.py`、`train/losses.py`、`train/rl_trainer.py` |
| 10 | **模型结构检查讲透**（第 6 章）：有 CPU 可运行的代码打印模块树、统计参数、选出 LoRA 目标并断言 lm_head 冻结；LoRA 配置照抄 YAML；记录里没有的原因按 Q1–Q4 的规则写并标注类别 | CI 跑 `tests/test_ch06_model_structure.py`；人工对照 `configs/sft/qwen35_2b_lora.yaml`、`train/model_utils.py` |
| 11 | **三种格式都能用**：每次构建都出网页、EPUB、PDF；PDF 中文无缺字、字体全部嵌入、有目录和页码、公式和代码正常显示；mermaid 图在三种格式里都显示；Pages 网页顶部能下载 EPUB 和 PDF；Release 附 EPUB、PDF、离线网页包和 `SHA256SUMS.txt` | CI 跑 `python3 build.py`、`scripts/check_outputs.py`（字体嵌入、页码、中文可抽取、EPUB 图片）、`epubcheck`；审核时下载 CI artifact `book`，翻看 PDF 预览页（含图的页） |
| 12 | **图要多且真实**：第 1–15 章每章至少 2 张图；数据曲线由 `scripts/figures/` 脚本从源仓库真实数据画出，注明数据来源，CPU 可重跑，不编数 | `check_book.py` A6、A7；CI 重跑画图脚本；审核时抽查曲线上至少 3 个点与原始文件一致 |
| 13 | **作者问题 Q1–Q10 答得有依据**：每题有记录依据（路径 / 提交 / 日志）、常见做法（带链接的论文或官方文档）和结论，结论标明【记录事实】【读者运行所得】【作者事后分析】之一；事后分析不写成历史事实；能在 CPU 上验证的（Q1、Q4、Q6）有脚本和测试 | `check_book.py` A8 检查 `docs/author-answers.md` 结构和标注；人工逐题审核；CI 跑对应测试 |

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
- 模型规模（作者 2026-10-19 确认）：选 Qwen3.5 2B 是因为它**足够小**。4B 也试过：完成了 SFT，之后因成本太高停止（见第 14 章）。为什么选 Base 而不选 Instruct：作者没有留下记录，由写作者按 Q10 的规则分析，正文标注【作者事后分析】，不写成作者当时的决定。

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
- 坑：rollout 中训练了外来的 `<|endoftext|>` / eos 终止符（提交 `33f3cc5`）；官方 Base 只配了 `<|endoftext|>` 作 eos，vLLM 不停在 `<|im_end|>`，导致输出失控（R48，第 11 章展开）。
- 素材：`configs/chat_templates/qwen35_agent.jinja`、`train/chat_format.py`、`tests/test_chat_format.py`、R48。

### 第 6 章 训练前先看清模型：Qwen3.5-2B 的结构与 LoRA 选型（专题）

要写细：读者读完要能自己打印出模型结构、数清参数、选出 LoRA 目标模块，并且知道为什么这样选。

- **模型结构**：
  - 官方 `Qwen/Qwen3.5-2B-Base` 的类名是 `Qwen3_5ForConditionalGeneration`。
  - 混合注意力：Gated DeltaNet 线性注意力层与全注意力层按 3 : 1 排列。
  - 词表 248,320，tie embedding。
  - 素材：`docs/apexo/09_custom_training_design.md` §1 表、`docs/apexo/00_progress.md` §1.1。
  - 各类模块逐个讲清楚：
    - 全注意力 `q_proj/k_proj/v_proj/o_proj`；
    - Gated DeltaNet 的 `in_proj_qkv/in_proj_z/in_proj_a/in_proj_b/out_proj`，其中 a/b 是衰减门控；
    - MLP `gate_proj/up_proj/down_proj`；
    - embedding、各层 RMSNorm、`lm_head`（与 embedding 共享权重）。
  - 层数、隐藏维度、参数总量这些具体数字，源仓库**没有记录**。只能由示例代码从官方 `config.json` 现场算出，并注明"读者运行所得，非项目记录"；或者等作者确认。
- **检查方法与 CPU 代码**（`examples/ch06_model_structure.py` + `tests/test_ch06_model_structure.py`）：
  1. 用源仓库 `tests/conftest.py` 里的微型 Qwen3.5 配置构建模型：2 层，`layer_types=["linear_attention","full_attention"]`，`tie_word_embeddings=True`。在 CPU 上随机初始化，不联网，不下载权重。
  2. 打印模块树（`named_modules()`），按类型汇总。
  3. 统计每类参数量，以及总参数量和 embedding 参数量。
  4. 按 `train/model_utils.py` 的 `DEFAULT_LORA_TARGETS` 选出 LoRA 目标，套上 `peft.LoraConfig(r=64, lora_alpha=64, lora_dropout=0.0, bias="none")`，打印可训练参数量和占比。
  5. 断言三件事：`lm_head` / embedding 处于冻结状态；GDN 的 `in_proj_a`、`in_proj_b` 被覆盖；可训练参数量等于按公式 Σ r·(d_in + d_out) 手算的结果。
  6. 附一个可选脚本，联网读取官方 2B 的 `config.json`，把同样的统计用在真实规模上。这个脚本不进 CI，正文里引用它的输出时要注明"读者运行所得，非项目记录"。
- **LoRA 目标模块与超参数**：
  - 精确配置照抄 `configs/sft/qwen35_2b_lora.yaml` 与 `configs/rl/qwen35_2b.yaml`：`lora.r=64`、`alpha=64`、`dropout=0.0`、`target_modules` 共 12 个。
  - 4B 照抄 `configs/sft/qwen35_4b_lora.yaml`。
  - 代码：`train/model_utils.py` 的 `apply_lora`（`bias="none"`、`task_type="CAUSAL_LM"`）。
- **哪些冻结、哪些训练，为什么**：
  - 只训练 LoRA 增量，基座权重、embedding、norm、`lm_head` 全部冻结。
  - `lm_head` 冻结有代码层面的原因：`_assert_head_frozen` 里写着，fp32 的 head 副本只缓存一次（`train/model_utils.py`；R5、R18）。
  - 为什么用 LoRA 而不做全参：`09_custom_training_design.md` §1.3 设计取舍表（单卡 40GB）。
  - 为什么 GDN 门控 a/b 也要加 LoRA：§6.2 第二轮审核"LoRA 补上 GDN 门控 `in_proj_a`、`in_proj_b`"。
  - 测试证据：`tests/test_model_utils.py`，说明见 `docs/apexo/05_testing_plan.md`：LoRA 覆盖 GDN 投影且 lm_head 冻结。
- **本章必备小节**（`check_book.py` 检查）：`## 动机`、`## 模型结构`、`## 检查方法与代码`、`## LoRA 目标模块与超参数`、`## 冻结与训练的参数`、`## 坑与修复`（表头同训练阶段章节）、`## 设计理由与事后分析`、`## 本章出处`。
- **本章配图**（至少 4 张）：模块树（高亮 12 个 LoRA 目标，灰色标冻结的 embedding / norm / lm_head）；一层全注意力与一层 Gated DeltaNet 的结构图；3 : 1 层排布示意；参数量对比表或条形图（总参数 vs 可训练参数，数字来自示例脚本，标注【读者运行所得】）。
- **本章验收**：
  1. 示例在 CPU 上跑通，测试断言了上面 5 条；
  2. 每个配置值都能在源仓库找到原文；
  3. 记录里没有的原因按 Q1–Q4 的规则写进「设计理由与事后分析」，每条标注类别；
  4. 不写源仓库没有记录的层数或参数量，除非标明【读者运行所得】。
- **作者问题**：Q1–Q4，由写作者回答，见下文「作者问题 Q1–Q10」。

### 第 7 章 SFT：在 Colab 上把格式教会（10-11 ~ 10-13，round2）

1. 小步验证：5 步 smoke（2B/4B loss 下降数字），再做正式长度短 SFT 和续训测试。
2. 正式 SFT：2 个 epoch；2B 702 步约 7 小时 45 分，4B 688 步约 11 小时 45 分；loss 曲线与显存（2B 13.91 GiB、4B 21.64 GiB，torch allocated）。
3. 坑与解决：
   - Drive 上的权重是 79 字节的符号链接替身，`safetensors` 报 `header too large` → 本地重建链接；
   - Drive 挂载约 67 分钟后掉线，进程卡死 → 材料先落 VM 本地盘；
   - causal-conv1d 与 CUDA 版本不匹配 → 跳过，走 torch 实现；
   - 40GB 上预演被"38 GiB 逻辑显存门禁"拦下但其实没 OOM → 改成物理显存采样门禁。
- 素材：`result/round2/README.md`、`result/round2/logs/formal-sft.log`、`result/round2/results/formal_sft/metrics_*.jsonl`、`result/round2/results/memory_audit_20261013.md`、§6.6–§6.8、§6.21–§6.27、`train/sft_trainer.py`、`train/dataset.py`。
- 建议配图：两条 SFT loss 曲线（由 metrics_*.jsonl 画，脚本放本书 `scripts/`）。

### 第 8 章 算力去哪里找：Colab、GCP 与 Agent Platform（10-11 ~ 10-14）

- 方案几次变化及原因：去掉 Daytona、拆 Colab/GCP（PR #4）→ 直接 GCP VM（round3，计划了但没执行）→ Agent Platform CustomJob + 临时 CPU 沙箱 VM（round4–round8）。
- 40GB 还是 80GB：§6.22、§6.32–§6.33 的决策过程；最后定为单卡 A100 40GB、训练与推理串行、KV cache 可调。
- 配额：A100 80GB 训练配额为 0；普通 A100 配额申请被拒；最后用 Vertex Spot A100 配额（R 记录与 §6.45–§6.46）。
- round9 的四次失败申请：项目编号字段用错、e2 机型不支持某维护策略、vCPU 超全局配额、pd-balanced 计入 SSD 配额——每次怎么分类、怎么修预检；"提交前两次只读核对"规则的由来。
- 镜像构建：causal-conv1d 源码编译卡住被取消（R47）。
- 素材：`result/round3/`–`result/round9/` 各 README、`docs/apexo/07_gcp_colab_a100_training_plan.md`、`docs/apexo/10_agent_platform_plan.md`、`result/round10/scripts/preflight_fit.py`、`result/round8/costs-and-retention.json`。
- 花费（作者 2026-10-19 确认）：全项目总花费约 **200 美元，全部是 Google Cloud**。仓库没有账单明细，书里只写这个总数，不要拆分到各轮。
- 为什么去掉 Daytona（作者 2026-10-19 确认）：Google Cloud 上已经有能承载沙箱的 VM，不需要再用 Daytona。

### 第 9 章 RL 原理与实现：DPPO 怎么在一张卡上跑

- 一步 RL 的完整流程：抽 4 个任务 × 每个 8 个样本 → vLLM 多轮 rollout（训练器离卡）→ 沙箱里执行 → verifier 0/1 奖励 → centered advantage → DPPO（binary TV 0.1）更新 LoRA → 权重推回 vLLM。
- 关键设计：零方差组过滤与有界 active sampling、截断样本 mask、`sync/logprob_abs_diff` 同步检查、k3 KL 只监控不惩罚（beta=0）、fp32 lm_head、分块 logprob 防 248K 词表 OOM、vLLM sleep/wake。
- 素材：`train/rl_trainer.py`、`train/rollout.py`、`train/losses.py`、`train/agent_env.py`、`configs/rl/qwen35_2b.yaml`、`docs/apexo/09_custom_training_design.md`、`docs/apexo/08_execution_contract.md`、`tests/test_rl_step.py`、`tests/test_losses.py`、提交 `1037af0`、`a5a8046`。

### 第 10 章 沙箱：让模型的命令只在该在的地方跑

- 为什么模型命令绝不能在宿主机裸跑；GPU worker 不跑 Docker；临时 CPU VM 跑原任务容器，经 IAP SSH 调用；VM 无服务账户、容器拿不到凭据（R44）。
- round5 原生 Shell Sandbox 实测（能力和限制），round9 原任务 verifier 探针（只证明链路通，不证明正确解法得 1 分）。
- 素材：`train/sandbox.py`、`train/remote_sandbox.py`、`scripts/agent_platform/cpu_startup.sh`、`result/round5/`、`result/round9/`、R40、R44。

### 第 11 章 评测：Harbor 与 Terminal-Bench 2.0 的长跑（10-14 ~ 10-17）

- 指标：89 个任务 × 5 次 = 445 trials；pass@1/3/5、valid_tool_call_rate。
- 指标怎么算、门禁怎么判，统一放到第 13 章；本章只讲评测怎么跑起来，以及跑的过程中出了什么问题。
- Base 评测前三次失败：24576 上下文装不下 16384 输出 → 65536 被 4 小时超时掐断 → 18 小时那次 SSH 断线退出 255。
- 根因与修复：R48（eos 停止符 + `repetition_penalty=1.05` + `max_tokens=8192` + `command_timeout=300`）、R49（Harbor 后台脱机运行、隧道重连、磁盘清理、trial 级快照与跨作业恢复）。
- Spot 抢占：SFT 评测 289 → 377 → 445 三次才跑完。
- 耗时：Base v6 约 13 小时 43 分；SFT 评测三次的时长。
- 素材：`result/round10/README.md`、`result/round10/conclusion.md`（上半部分）、`result/round10/harbor-base/`、`result/round10/harbor-sft-2b/`、`scripts/agent_platform/harbor_eval.py`、`evaluation/run_eval.py`、R48、R49、§6.49–§6.50。

### 第 12 章 RL 实跑：八步，以及前两步为什么全是 0 分（10-17 ~ 10-18）

- Step 1 后被 4 小时超时取消 → 改 18 小时，从已提交 checkpoint 续训；Step 1–2 奖励全 0、梯度跳过；继续到 Step 8，5 步有非零奖励更新。
- 逐步指标表（reward_mean、useful_groups、loss、KL、entropy、显存、步时），步时 3,500–8,700 秒。
- 素材：`result/round10/rl-2b/`（`continue-final/metrics.jsonl`、`rollouts.jsonl`、各 run log）、`result/round10/sub2-orchestrator-status.json`、`result/round10/scripts/sub2_orchestrate.py`。

### 第 13 章 怎么判断成功还是失败：指标、算法与门禁（专题）

要写细：每个信号都要讲清楚四件事——看什么、怎么算（公式和样本量）、为什么这样算、证据在哪个文件。全部对应到源仓库的门禁脚本和配置。

- **评测信号**（`evaluation/run_eval.py`）：
  - 样本：Terminal-Bench 2.0 的 89 个任务 × 每任务 5 次 = 445 trials。一个 trial 的 reward 取 `verifier_result.rewards.reward`，reward ≥ 1 记为通过。没有 reward 记 0 分，同时记一次 error。
  - **pass@1**：先算每个任务的通过率 c/n（n=5），再对 89 个任务取平均（按任务等权的宏平均）。
  - **pass@k**：无偏估计 `1 − C(n−c, k) / C(n, k)`（Chen et al. 2021），对任务取平均；k ∈ {1, 3, 5}。
  - **valid_tool_call_rate**：`1 − 格式错误步数 / 总步数`，在全部 trial 上合并计算（按步汇总）。格式错误指 Harbor `timing.json` 里 `format_error: true` 的步，也就是这一步没有合法的 bash 调用。
  - **置信区间**：配对 bootstrap，函数 `paired_bootstrap`。
    1. 取两个报告共有的任务，逐任务求差 d_t = 当前通过率 − 对照通过率；
    2. 有放回重采样 89 个差值，求均值，重复 2,000 次，随机种子 0；
    3. 排序后取第 2.5 和第 97.5 百分位作为 95% 区间（代码里取排序后下标 50 和 1950 的值，下标从 0 开始）。
    - 为什么配对：同一任务的难度在两次评测之间是共享的，配对能消掉任务难度带来的方差。为什么选 2,000 次、为什么用种子 0，记录里没有写，按 Q6 处理。
- **训练信号**：
  - SFT：loss 曲线与 grad_norm（`result/round2/results/formal_sft/metrics_*.jsonl`；`max_grad_norm=1.0`）。
  - RL（`result/round10/rl-2b/continue-final/metrics.jsonl`，字段定义见 `docs/apexo/08_execution_contract.md`）：
    - `rollout/reward_mean`、`useful_groups`（零方差组被过滤，见 `train/rl_trainer.py` 的 `_useful_groups`）、`train/skipped`；
    - `train/loss`、`train/kept_frac`、`train/tv_mean`（DPPO binary TV 0.1）；
    - `sync/logprob_abs_diff`：训练端与 vLLM 的 logprob 一致性；
    - `rollout/truncated_frac`、`rollout/format_errors_mean`、`done/no_tool_call`。
  - **KL**：k3 估计 `exp(d) − d − 1`，其中 `d = log π_ref − log π`，截断在 [−20, 20]（`train/losses.py` 的 `k3_kl`）。参考模型就是"关闭 LoRA 的同一个模型"，每 `algo.kl_monitor_every=10` 步算一次；beta=0，只监控、不惩罚（`09_custom_training_design.md`、R9）。
  - **熵**：`train/entropy`。监控时看是否"> 2× 初值"（`05_testing_plan.md` 的监控表），门禁用的是"末值 ≤ 3× 初值"（`run_eval.py`）。两个数都照写，不要合并成一个。
  - **格式塌陷**：最近 10 个训练行的 `done/no_tool_call / rollout/samples` 取平均，必须 ≤ 10%。这个阈值在代码里写死，不随有效调用率阈值变化（`rl_training_checks` 的注释；R8）。
  - **重复与停止符问题**：R48。官方 Base 只把 `<|endoftext|>` 配成 eos，vLLM 不会停在 `<|im_end|>`，输出就会失控。修复是加停止符、`repetition_penalty=1.05`、`max_tokens=8192`、`command_timeout=300`（`scripts/agent_platform/harbor_eval.py`）。另见提交 `33f3cc5`。
- **门禁与判定**：
  - 原阈值：SFT 与 RL 都是有效调用率 ≥ 98%，pass@1 增益 ≥ +0.02 且 CI 下界 > 0。
  - 修订阈值（10-19，Apexo 阈值）：
    - SFT：≥ 96%，增益 ≥ +0.01 且 CI 下界 > 0；
    - RL：≥ 96%，不劣于 −0.01；
    - KL ≤ 0.2、熵 ≤ 3×、无工具调用 ≤ 10% 不变。
  - 两套阈值下的结论都写：原阈值下 FAILED，作者人工放行（`sft-user-approved.json`、`rl-user-approved.json`）；修订阈值下 PASSED（`sft-gate-apexo.json`、`rl-gate-apexo.json`、`result/round10/regate/README.md`）。
  - 修订原因只写作者给的：作者决定按 Apexo 阈值判定。不写成训练前预设。
  - KL 0.2 在 `05_testing_plan.md` 里标着"暂定，T2 后标定"。记录里没有标定结果，按 Q7 处理。
- **实测值**：
  - Base / SFT / RL 的 pass@1、pass@3、pass@5、有效调用率，SFT 与 RL 的增益和 CI，kl_max 0.1979，无工具调用 8.23%。
  - 全部照抄 `result/round10/harbor-*/` 下的报告。
- **CPU 代码**（`examples/ch13_metrics.py` + `tests/test_ch13_metrics.py`）：
  - 实现 pass@k、有效调用率、配对 bootstrap 和 k3 KL；
  - 用一个虚构的小数据集演示，测试断言结果与公式一致、与 `evaluation/run_eval.py` 同样输入下的输出一致；
  - 附一个可选脚本，从源仓库报告重算增益和 CI，不进 CI。
- **本章必备小节**（`check_book.py` 检查）：`## 动机`、`## 信号总表`、`## 评测信号`、`## 训练信号`、`## 门禁与判定`、`## 实测值`、`## 设计理由与事后分析`、`## 本章出处`。
- **本章配图**（至少 4 张）：门禁判定流程图（SFT、RL 各一张，含原阈值与修订阈值两条分支）；RL 8 步的 KL 与 entropy 曲线（标出 0.2 上限、2× 告警线和 3× 门槛）；`done/no_tool_call` 占比曲线并标出最近 10 步窗口；配对 bootstrap 差值分布直方图（由真实逐任务结果重采样得到，标出 95% 区间）。
- 「信号总表」的表头固定为 `| 信号 | 公式 | 样本量 | 数据来源 | 原阈值 | 修订阈值 | 为什么这样算 | 证据 |`，每个信号一行。
- **本章验收**：
  1. 上面列出的每个信号都在总表里；
  2. 公式与源码逐字对应，并注明函数名；
  3. 样本量写全；
  4. 原阈值和修订阈值两套结论都写；
  5. 示例与测试通过；
  6. 记录里没有的理由按 Q5–Q9 的规则写进「设计理由与事后分析」，每条标注类别。
- **作者问题**：Q5–Q9，由写作者回答，见下文「作者问题 Q1–Q10」。

### 第 14 章 结果与诚实复盘

- 三次全量评测对比：pass@1 0.00% → 1.12% → 0.90%；valid_tool_call_rate 88.96% → 96.58% → 96.24%；各自解出了哪些任务。
- 门禁结论（Apexo 阈值）：SFT **PASSED**（有效调用率 96.58%；pass@1 增益 +0.0112，CI [+0.0022, +0.0247]）；RL **PASSED**（有效调用率 96.24%；pass@1 变化 −0.0022，CI [−0.0112, +0.0067]；kl_max 0.1979；无工具调用 8.23%）。素材：`result/round10/harbor-sft-2b/sft-gate-apexo.json`、`result/round10/harbor-rl-2b/rl-gate-apexo.json`。
- 讲清楚门禁通过能说明什么、不能说明什么：RL 只是"没有明显变差"，不是"变好了"。
- 只做了 8 步 RL、2B 单卡，结论边界；本项目不和任何外部报告的数字做比较。
- 4B 为什么停（作者 2026-10-19 确认）：**成本太高**。4B 完成了 SFT（第 7 章），之后没有继续 RL 和全量评测。
- 素材：`result/round10/conclusion.md`、`result/round10/harbor-*/`、§3。
- 如果重来（作者 2026-10-19 回答）：大概会直接沿用现有方法，只是预算会更多。

### 第 15 章 工程纪律：让这个项目可复盘的那些规矩

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

## 作者问题 Q1–Q10（由写作者回答，Grok Bot 审核）

作者 2026-10-19 20:44 SGT 指定：这 10 个问题由写作者（dot）分析和回答，Grok Bot 逐条审核。答案单独开 PR：新建 `docs/author-answers.md`，并把结论写进对应章节（第 2、6、13 章的「设计理由与事后分析」或正文）。**截止 2026-10-21 18:00 SGT，与第 1 批一起提交。**

`docs/author-answers.md` 格式（`check_book.py` A8 检查）：每题一个二级标题 `## Q1 ……`，下面三个三级小节：

| 小节 | 要写什么 |
| --- | --- |
| `### 记录依据` | 源仓库里能找到的全部相关记录：文件路径、提交号、日志行、配置值。找不到就写"源仓库没有相关记录"，并列出查过的位置 |
| `### 常见做法与出处` | 业界常见做法和理由，**每条带可打开的链接**（论文、官方文档、库的源码或 README）。不许引用打不开或不存在的来源 |
| `### 结论` | 第一行是类别标签：`【记录事实】`、`【读者运行所得】` 或 `【作者事后分析】`（可以组合，每句话归到一类）；然后是写进书里的说法 |

逐题要求：

| # | 问题 | 记录依据从哪里找 | 必须做的计算 / 要点 |
| --- | --- | --- | --- |
| Q1 | 训练前是怎么查看模型结构的 | `docs/apexo/00_progress.md` §1.1、`09_custom_training_design.md` §1、`train/model_utils.py`（`DEFAULT_LORA_TARGETS`、`HF_TEXT_PREFIX` / `OFFICIAL_TEXT_PREFIX`）、`tests/conftest.py`、`tests/test_model_utils.py`、相关提交 | 写出可复现的检查流程；`examples/ch06_model_structure.py` 打印模块树并按类型汇总（CPU、不下载权重），`tests/test_ch06_model_structure.py` 断言 |
| Q2 | 为什么 LoRA r=64、alpha=64、dropout=0 | `configs/sft/qwen35_2b_lora.yaml`、`configs/rl/qwen35_2b.yaml`、`configs/sft/qwen35_4b_lora.yaml`、`09_custom_training_design.md`、提交历史里这几个值有没有改过（`git log -S`） | 讲清 alpha/r 缩放（这里缩放系数为 1）、r 与可训练参数量的关系、dropout 在 RL 里的影响；引用 LoRA 原论文与 PEFT 文档 |
| Q3 | 为什么冻结 embedding 和 norm | `train/model_utils.py`（`apply_lora`、`_assert_head_frozen`）、R5、R18、tie embedding 的事实 | 讲清 tie embedding 下训练 embedding 等于同时改 lm_head、248,320 词表的参数和显存代价、norm 参数量很小；记录里的代码原因标【记录事实】，其余标【作者事后分析】 |
| Q4 | 当时有没有记录总参数量和可训练参数量 | 在源仓库全文搜索参数量相关字段（日志里 PEFT 的 `trainable params`、`print_trainable_parameters` 等），写明查过哪些文件 | 没有记录就如实写；另用示例脚本从官方 `Qwen/Qwen3.5-2B-Base` 的 `config.json`（复制进本书 `data/`，注明 revision）在 meta device 上算出总参数量和 r=64 下 12 个目标模块的可训练参数量，并与公式 Σ r·(d_in + d_out) 手算结果比对，测试断言两者相等；标【读者运行所得】 |
| Q5 | pass@1 为什么按任务平均，而有效调用率按步合并 | `evaluation/run_eval.py`（pass@k 和 `valid_tool_call_rate` 的实现）、`05_testing_plan.md` | 用真实数据说明两种口径的差别：同样的数据按任务平均和按步合并各算一遍；引用 pass@k 无偏估计（Chen et al. 2021）和 macro / micro 平均的常见用法 |
| Q6 | 置信区间为什么用 2,000 次重采样、种子 0 | `evaluation/run_eval.py` `paired_bootstrap`（`iters=2000, seed=0`，区间取排序后下标 50 和 1950） | 写脚本 + 测试：用真实逐任务通过率（从 `result/round10/harbor-*/` 报告复制到 `data/`），换 10 个以上种子、iters ∈ {1000, 2000, 10000} 重算 SFT、RL 两个 CI，报告端点波动范围，说明结论（CI 下界是否 > 0 等）稳不稳；引用 bootstrap 文献（Efron & Tibshirani）或 SciPy `bootstrap` 文档 |
| Q7 | KL 上限 0.2 标为"暂定"，后来校准过吗 | `05_testing_plan.md`（"暂定，T2 后标定"）、`run_eval.py`（`--kl-max` help 文本 "provisional"）、`result/round10/rl-2b/continue-final/metrics.jsonl` 的 KL 序列、提交历史 | 查清有没有标定记录；没有就写"没有标定记录"（【记录事实】）；把实测 kl_max 0.1979 与 0.2 的距离画出来；k3 估计引用 Schulman 的 KL 近似说明 |
| Q8 | 熵为什么告警 2×、门槛 3× | `05_testing_plan.md` 监控表（> 2× 初值告警）、`run_eval.py`（末值 > 3× 初值失败）、RL 实测 entropy 序列 | 两个数分别出自哪里；为什么告警线应低于门槛（先预警再判失败）标【作者事后分析】；画出实测 entropy 曲线和两条线 |
| Q9 | 格式塌陷为什么只看最后 10 步 | `run_eval.py` `rl_training_checks`（`train_rows[-10:]`、`no_tool_limit = 0.10`、只统计含 `rollout/samples` 的行）、R8 | 用实测 8 步数据说明：训练行少于 10 时窗口就是全部训练行；对比"全程平均"和"最近 N 步"在本项目数据上的结果 |
| Q10 | 为什么用 Base 而不是 Instruct | `AGENTS.md`、`docs/apexo/01_architecture.md`、`09_custom_training_design.md`、R48（Base 的 eos 配置）、提交历史 | 记录里没有就写明；常见做法引用 Qwen 官方模型卡 / 技术报告里 Base 与 Instruct 的区别；结论标【作者事后分析】，也写出用 Base 带来的实际问题（R48 停止符） |

审核规则：任何一处把事后分析写成"作者当时这样决定"、引用打不开的链接、或数字找不到原文，整题退回。

## 已知记录缺口

作者问题 Q1–Q10：由写作者回答，见上一节。

已解决（作者 2026-10-19 回答）：
- 实际花费：约 200 美元，全部是 Google Cloud。
- 4B 停止原因：成本太高。
- 方法归属：Apexo，作者自行设计。
- 门禁阈值：Apexo 阈值，见第 13 章。
- 源仓库引用方式：保持私有，用纯文本写路径和提交号。
- 为什么去掉 Daytona：GCP 上已有能承载沙箱的 VM。
- 为什么选 2B：足够小。
- 如果重来：沿用现有方法，加大预算。

记录说明（不是待问问题）：RL 评测 Attempt 1 的原始台账和日志已被 Attempt 2 覆盖。Drive 上只留下一份较早的编排状态快照（316 trials 时点），已收进源仓库 `result/round10/harbor-rl-2b/attempt1-orchestrator-status-drive.json`。书里照此写，不要补细节。

## 交付批次

| 批次 | 章节 |
| --- | --- |
| 1 | 第 0、1 章；作者问题 Q1–Q10 的答案（单独 PR，2026-10-21 18:00 SGT 前） |
| 2 | 第 2、3、4 章 |
| 3 | 第 5、6、7 章 |
| 4 | 第 8、9、10 章 |
| 5 | 第 11、12、13 章 |
| 6 | 第 14、15 章与附录 |
