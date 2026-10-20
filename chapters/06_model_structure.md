# 第 6 章 训练前先看清模型：Qwen3.5 的结构与 LoRA 选型

## 动机

第 3 章的审核发现，写上“LoRA”并不足以说明适配器装对了。Qwen3.5 不只有常见的 q/k/v/o 投影，还混合了 Gated DeltaNet；如果只凭旧模型的模块名单套配置，某些门控投影可能没有被覆盖。另一方面，权重文件里的名字、文本训练模型里的名字和推理端的名字也不相同。

本章做一件具体的事：在 CPU 上创建随机微型模型，打印真实模块树，数清唯一参数，列出每个 LoRA 命中点，并用矩阵形状核验可训练参数量。它不加载正式权重，不证明预训练能力，也不复现原项目的 GPU 训练。读者首先应当能回答“改的是哪块矩阵”，再去讨论更大的性能问题。

项目事实固定于 `ca57456`。已记录的设计理由与本书实验结果分开标记；未记录的作者动机保持空缺，不为了填满一个小节而编造故事。

## 模型结构

### 6.1 官方包装、文本骨干与 checkpoint 名字

源设计文档记录官方架构名为 `Qwen3_5ForConditionalGeneration`，包含文本和视觉包装。项目的文本训练路径使用 `Qwen3_5ForCausalLM`，训练只需终端任务的文本序列，不据此宣布官方模型没有视觉结构。

源 `train/model_utils.py` L36–37 区分文本权重前缀：训练端的 `model.` 与官方布局的 `model.language_model.`。导出与推理同步需要相应映射。名字能对上只是必要条件，形状、数值和接收端是否真正接受权重仍需测试。`tests/test_model_utils.py` 用随机微型官方布局 checkpoint 检查这些映射，而不是只看输出目录存在。

本书实验直接从文本配置构造 CausalLM，省去下载、保存、重新加载视觉包装的环节。它沿用原项目 fixture 的文本维度，不把自己的路径称为原项目测试完整复现。

### 6.2 两种注意力层和共同的 MLP

源设计记录的真实架构排列是线性注意力与全注意力 **3∶1**。这是一种层类型排列，不是 75% 参数属于线性注意力，也不是延迟必然降低 75%。每类层的参数大小、激活和执行成本另有口径。

![图 6-1　架构中的 3∶1 层类型示意；本章 CPU fixture 只取一层线性与一层全注意力。](images/ch06_layer_pattern.svg)

全注意力层有 `q_proj`、`k_proj`、`v_proj`、`o_proj`。它们分别形成查询、键、值和输出投影。不要从名称猜测矩阵一定都是正方形：本章微型配置里，`q_proj` 的输出维度是 128，而隐藏维度是 64；应读取模块的 `in_features`、`out_features`，不要把所有投影统一写成 64×64。

Gated DeltaNet 路径包括 `in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b` 与 `out_proj`。q/k/v 来自合并投影，卷积、递推状态、门控和输出投影共同完成这一层的计算。a 与 b 都是需要检查的门控相关投影，但不是同一个门：固定 Transformers 5.6.0 实现里，b 经 sigmoid 形成更新强度 beta，a 进入衰减量的计算。源项目的目标列表注释把它们统称 decay gates，本书保留模块名，但不把这句简称当作二者数学作用完全相同的证据。

两种层后面都有 MLP 的 `gate_proj`、`up_proj`、`down_proj`。门控分支与上投影相互作用，再降回隐藏维度；这里的 `gate_proj` 也不是 GDN 的 `in_proj_a/b`。名称中都带“门”，不表示可互相替代。

![图 6-2　全注意力与 Gated DeltaNet 的主要投影路径；共同接到 MLP，图仅表达职责而非完整方程。](images/ch06_attention.svg)

技术定义可参阅 [Qwen3.5 官方集成文档](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5)；本书运行固定 Transformers 5.6.0，a/b 的说明核对该版本 [modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，不把未来文档的新增优化当成本次运行配置。

### 6.3 embedding、norm 与输出头

embedding 把 token ID 映射成隐藏向量；norm 调整层内数值尺度；`lm_head` 把最终隐藏向量映射回词表分数。源项目词表为 248,320，配置使用共享输入／输出权重，也就是 tie embedding。

共享不是“两个数值相同的副本”。在本章实例中，输入 embedding 与输出 head 的 weight 是同一个 Parameter 对象。按模块分别相加会重复计数；`parameters()` 的去重遍历才是本章“唯一参数量”的分母。与此同时，推理时输出头仍要做计算，冻结或共享权重并不会让词表投影成本消失。

## 检查方法与代码

### 6.4 不下载模型的完整实验

在第 1 章固定 CPU 依赖环境运行：

```bash
python examples/ch06_model_structure.py --json /tmp/ch06-reader.json
pytest -q tests/test_ch06_model_structure.py
```

程序按以下顺序完成检查：

1. 用 `Qwen3_5TextConfig` 构造两层随机文本模型。一层 `linear_attention`，一层 `full_attention`；hidden=64、intermediate=128、词表=248320、共享 embedding。它来自源 `tests/conftest.py` 的微型文本配置，不是完整 2B 的层数和宽度。
2. 遍历 `named_modules()`，打印名字与类型，汇总模块类型计数。该 API 的意义见 [PyTorch Module 文档](https://docs.pytorch.org/docs/2.10/generated/torch.nn.Module.html)。
3. 在安装适配器前记录唯一基座参数和 embedding 参数；按 12 个目标叶名称找到实际 `Linear` 模块，读取每个矩阵的输入／输出维度。
4. 安装 PEFT LoRA：r=64、alpha=64、dropout=0、bias=none、CAUSAL_LM。只训练低秩增量，不训练基座。
5. 比较实测可训练参数总数与每个命中矩阵的手算和，断言 embedding/head 冻结、权重仍共享、所有可训练参数都属于 LoRA，a/b 均被命中。

运行时可能提示没有 fast path 而回退到 torch 实现，这是 CPU 教学环境的预期边界，不要求安装 GPU Triton 内核。例子只做结构与参数检查，没有吞吐或显存测量；不要把“几秒完成结构检查”当作正式模型推理速度。

![图 6-3　微型文本模型的模块分组；绿色是 12 类 LoRA 目标，灰色组件保持冻结。](images/ch06_module_tree.svg)

### 6.5 怎样读这次输出

下面全部标为 **【读者运行所得】**。本书固定教学环境实际生成 `data/ch06_tiny_parameters.json`，CI 重新构造并核对；它不是原训练项目当年的参数统计。

| 项目 | 本章两层微型文本模型的输出 | 口径 |
| --- | --- | --- |
| 唯一基座参数 | 15,975,800 | 安装 LoRA 前；共享 head 不重复计数 |
| embedding 唯一参数 | 15,892,480 | 248,320 × 64，已包含在上一行 |
| LoRA 可训练参数 | 143,872 | 15 个投影实例的低秩增量之和 |
| 安装适配器后的唯一参数 | 16,119,672 | 基座 + LoRA，不是二者取平均 |
| 可训练占比 | 约 0.8925% | 143,872 / 16,119,672 |
| 目标叶名称／实际命中实例 | 12 类／15 个 | 两层 MLP 都包含三个目标名称 |

“12 个 target_modules”说的是 12 种叶名称，不是完整网络只有 12 个适配器。一个名称在多层出现会命中多个实例。反过来，只检查总数可能漏掉错误分配：少装一块、多装另一块，参数量可能碰巧接近。因此本书同时检查目标集合、逐实例形状和总和。

![图 6-4　两层微型模型的唯一参数与可训练参数，均为读者 CPU 实验所得；不是完整 2B 模型统计。](images/plots/ch06_parameters.png)

### 6.6 可选的真实规模配置检查

`examples/ch06_official_config_optional.py` 是单独的可选联网入口，真实联网／模型构造不进 CI；CI 只用假下载器检查它的固定 revision 与 config-only 合同。本批未运行它，也没有在正文填写其输出。需要时由读者显式运行：

```bash
python -m examples.ch06_official_config_optional
```

该入口只下载官方 `Qwen/Qwen3.5-2B-Base` 在 revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` 的 `config.json`，在 meta device 上构造文本模型形状，不下载权重、不创建 GPU。输出明确只统计文本骨干和共享输出头，排除视觉塔；meta 张量没有真实数值，不能据此测试 forward 正确性。任何从这个入口取得的数字都必须标为读者运行所得，不能回填成项目日志曾经记录过的数字。

## LoRA 目标模块与超参数

### 6.7 配置值与矩阵公式分别核对

源 2B SFT、2B RL 与 4B SFT YAML 均配置 r=64、alpha=64、dropout=0.0；目标为全注意力四类、GDN 五类、MLP 三类。`apply_lora` 另明确 `bias="none"`、`task_type="CAUSAL_LM"`。这些是项目配置事实，不是本章搜索得到的最优超参数。

对一个输入宽度为 d_in、输出宽度为 d_out 的无偏置线性层，LoRA 新增 A 的 r×d_in 和 B 的 d_out×r，因此新增量为 r(d_in+d_out)。本章的缩放 alpha/r=1。缩放等于 1 不表示更新恒为 1，也不表示 rank 不再影响可训练空间。LoRA 的技术归属是 Hu 等的[原论文](https://arxiv.org/abs/2106.09685)，PEFT 的配置接口见[官方 LoRA 文档](https://huggingface.co/docs/peft/main/en/package_reference/lora)。

在本章 fixture 中可逐项手算：

| 分组 | 实例形状与数量 | r=64 的新增参数 |
| --- | --- | --- |
| GDN | qkv:64→128；z、out:64→64；a、b:64→4 | 12,288 + 8,192×2 + 4,352×2 = 37,376 |
| 全注意力 | q:64→128；k、v:64→32；o:64→64 | 12,288 + 6,144×2 + 8,192 = 32,768 |
| 两层 MLP | 每层 gate/up:64→128，down:128→64 | 12,288×6 = 73,728 |
| 合计 | 15 个线性投影实例 | 143,872 |

r=64 在这个微型模型里并不一定比每块原矩阵都“低秩节省”，尤其 a/b 输出只有 4。此例故意保留项目 r 值以验证接口和公式，不能拿微型网络上每块增量／基座比例来证明真实规模的经济性。

## 冻结与训练的参数

LoRA 安装后，基座矩阵、embedding、norm 与输出头全部保持冻结；训练集合只含 A/B。这里的“冻结”指 `requires_grad=False`，不等于在 forward 中完全绕开这些模块。

项目还对输出头提出一个实现约束。以下是 `train/model_utils.py` L119–122，提交 `ca57456` 的连续静态摘录，原样保留，只供阅读：

```{.python .source-excerpt}
def _assert_head_frozen(peft_model) -> None:
    _, head = get_backbone_and_head(peft_model)
    if head.weight.requires_grad:
        raise RuntimeError("lm_head must stay frozen under LoRA (the fp32 head copy is cached once)")
```

【记录事实】冻结 head 不只是参数高效训练的一般习惯：实现会缓存一次 fp32 head 副本。如果原 head 之后还被优化器改变，副本可能过时。输入 embedding 与 head 共享时，也不能只检查名字叫 `lm_head` 的参数而忽略别名。风险 R5、R18 及测试补上了这条边界。

## 坑与修复

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 只覆盖传统 q/k/v/o，漏掉 GDN 门控投影 | 将旧架构的目标名单直接迁移 | 第二轮审核补上 in_proj_a、in_proj_b；核对实际叶名称集合 | `docs/apexo/00_progress.md` §6.2；`train/model_utils.py`，`ca57456` |
| 权重名字存在，却没有正确同步到官方／推理布局 | 文本、官方包装及推理端的前缀不同 | 显式映射并检查接收端报告，失败不当作成功 | `tests/test_model_utils.py`；风险 R4、R11，`ca57456` |
| 输出头仍可训练，fp32 副本只建一次 | 缓存与优化器状态假设冲突 | 加冻结断言，LoRA加载后也检查 | `_assert_head_frozen`；R5、R18，`ca57456` |
| 把 CPU fixture 的维度写成完整2B结构 | 混淆测试模型与正式配置 | 所有两层/64宽度/参数量标为读者实验，真实规模只在可选入口检查 | `tests/conftest.py` 与本章例子 |
| 将共享 head 与 embedding 重复累加 | 按模块名统计而非唯一 Parameter | 使用去重参数遍历，另检查对象共享 | 本章测试与 PyTorch Module API |

## 设计理由与事后分析

【记录事实】源 `09_custom_training_design.md` §1.3 将 LoRA 与单卡 40GB 预算联系起来；`00_progress.md` §6.2 记录补齐 GDN 门控目标；head 冻结有缓存一致性的代码理由。它们可以说明项目做过什么，以及记录里明确写了什么约束。

【读者运行所得】本章验证了目标集合、形状、冻结状态和参数公式。公式不等于作者曾经按这套步骤决策；143,872 不是项目正式2B的可训练参数量。

没有可核验的选择过程，就不声称 r64、alpha64、dropout0 是经消融最优，也不替作者补写每个 norm 冻结的心理动机。本章止于已记录的理由与既定结构实验，不把额外事后分析包装成历史事实。

## 本章出处

- `docs/apexo/09_custom_training_design.md` §1、§1.3：架构、词表、共享 embedding 与已记录设计取舍；`00_progress.md` §1.1、§6.2。提交 `ca57456`。
- `train/model_utils.py` 的目标列表、前缀映射、`apply_lora` 及 L119–122；`tests/test_model_utils.py`、`tests/conftest.py`、`docs/apexo/05_testing_plan.md`。提交 `ca57456`。
- `configs/sft/qwen35_2b_lora.yaml`、`configs/rl/qwen35_2b.yaml`、`configs/sft/qwen35_4b_lora.yaml`；风险登记 R4、R5、R11、R18。提交 `ca57456`。
- Transformers 5.6.0 的 Qwen3.5 实现、PEFT 0.18.1；本书 `examples/ch06_model_structure.py`、`data/ch06_tiny_parameters.json` 与对应测试。参数数字为本书 CPU 读者实验。
