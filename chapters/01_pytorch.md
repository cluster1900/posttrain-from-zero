# 第 1 章 PyTorch 入门：读懂后训练代码需要的那部分

## 1.0 本章的任务和运行方式

读后训练代码最容易遇到的障碍，不是看不懂某个 Python 语句，而是不知道张量代表什么。一个维度是样本、token，还是词表？loss 是一条轨迹的均值，还是整个更新步的均值？`checkpoint` 指保存到磁盘，还是反向时重算激活？这些差别会直接改变训练结果。

本章先用小例子建立心智模型，再对照真实代码。所有教学数值均为人为构造，与项目实测数字无关。配套程序在本书 `examples/ch01_pytorch.py`；在仓库根目录按 `requirements-examples.txt` 安装 CPU 教学依赖后运行：

```bash
python3 examples/ch01_pytorch.py
```

程序不访问模型 Hub，不需要私有数据，也不需要 GPU。每节对应一组断言，而不是只打印一个看起来合理的 loss。正文中六段 Python 代码原样汇集于 `examples/ch01_snippets.py`；测试同时检查逐字一致与执行结果。项目的 GPU 环境另有一套锁定依赖：`torch==2.10.0`、`transformers==5.6.0`、`peft==0.18.1`、`vllm==0.19.1`。这是源仓库 `requirements.txt`（提交 `ca57456`）的记录，不是让 CPU 入门例子安装整个推理栈。运行环境与验证结果见本书 `docs/batch1-validation.md`。

## 1.1 张量：先看形状，再看精度和设备

### 不把 token ID 当成连续数值

张量是带形状、数据类型和设备位置的数组。对语言模型，常见输入 `input_ids` 的形状是 `[B, T]`：`B` 为批大小，`T` 为序列长度；每个整数是词表索引。经过 embedding 后通常得到 `[B, T, H]`，`H` 为隐藏维度。最后的词表分数 logits 通常为 `[B, T, V]`，`V` 为词表大小。这里只是在约定字母含义，不要求所有实现都同时物化这些张量。

```python
import torch
ids = torch.tensor([[1, 3, 2]], dtype=torch.long)
embedding = torch.nn.Embedding(8, 4)
hidden = embedding(ids)
assert ids.shape == (1, 3)
assert hidden.shape == (1, 3, 4)
assert ids.device.type == "cpu"
```

索引用整数；可学习权重和中间表示通常用浮点数。把 token ID 转成 bf16 并不会“省显存地输入模型”，反而破坏了索引接口。相反，`hidden.to(torch.bfloat16)` 是改变浮点存储精度。`.to()` 通常需要接住返回值，不要以为写一行 `hidden.to(...)` 就一定修改了原对象。

### fp32、bf16 与“省了一半”的边界

fp32 每个元素 4 字节，bf16 每个元素 2 字节。bf16 保留较大的指数范围，但尾数精度比 fp32 低。它不是“计算结果完全相同、只是更小”的格式；小差值、概率计算和累加中的误差仍需注意。CPU 教学只需观察转换和字节数，不用 CPU 的执行时间判断 A100 上的收益。

对一个不共享存储的普通稠密张量，`numel() * element_size()` 能算出元素存储字节数。它不包含 Python 对象、分配器缓存或其他张量，也不能直接代表进程峰值内存。

```python
import torch
x = torch.zeros(2, 3, dtype=torch.float32)
y = x.to(torch.bfloat16)
assert x.numel() * x.element_size() == 24
assert y.numel() * y.element_size() == 12
```

训练时至少要分清四类占用：权重、梯度、优化器状态、前向激活。AdamW 为被更新的参数维护状态；长序列还会让中间激活和临时算子结果变大。LoRA 冻结基座后，可减少基座的梯度和优化器状态，但基座权重仍要存在，反向传播所需的激活也不会凭空消失。

源项目并非全程一种精度：`model.dtype=bfloat16`，同时 `lm_head_fp32=true`。`train/model_utils.py` 中 `frozen_head_weight` 为冻结的输出头准备 fp32 副本；`train/losses.py` 分块计算 token log-probability。换言之，要按实际张量生命周期记账，不能只用“参数量 × 2 字节”估计训练显存。（源配置与代码，提交 `ca57456`。）

**这一节的检查点：**给你一个 `[1, 100, 8]` 的浮点张量，你应能解释三个维度的约定，并知道“8”究竟是隐藏维度还是词表维度，要从产生它的代码判断。

## 1.2 自动微分与一次真正的参数更新

### backward 不会替你更新参数

把训练想成四步：前向计算预测，损失函数衡量误差，反向计算参数的导数，优化器利用导数更新参数。PyTorch 会跟踪参与求导的运算；调用 `backward()` 后，叶子参数的 `.grad` 存放梯度。图中的中间结果用于链式求导，不等于每个中间张量都会自动保留可读的 `.grad`。

```python
import torch
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w], lr=0.1)
optimizer.zero_grad(set_to_none=True)
loss = (w - 2.0).square()
loss.backward()
assert w.item() == 0.0                 # 仍未更新
assert w.grad.item() == -4.0
optimizer.step()
assert torch.isclose(w.detach(), torch.tensor(0.4))
```

这是一个一维教学问题，不是语言模型 loss。最重要的现象是：`backward()` 与 `step()` 分工不同。另一个常见错误是忘记清梯度。PyTorch 默认把新梯度累加进 `.grad`，并不在每次反向之前自动归零。这个行为既可能造成 bug，也正是梯度累积的基础。[PyTorch 自动微分说明](https://docs.pytorch.org/docs/2.10/notes/autograd.html)

`model.eval()` 与 `torch.no_grad()` 也不能混用：前者切换 dropout 等模块的训练行为，后者关闭梯度记录。训练 LoRA 时，冻结的基座参数不求自己的梯度，但计算链仍需把梯度传回前面的可训练适配器，不能把整个模型前向包进 `no_grad()`。

### 为什么 bs=1 还能有全局 batch 16

显存装不下多条长序列，可以逐条前向、逐条反向，让梯度留在参数上，最后统一更新一次。源项目 SFT 的单个 micro-step 只处理一条序列，不做 padding 或 packing；配置中的 `global_batch_size=16` 是每次优化器更新覆盖的样本数。在单设备完整批次下，这对应 16 次 micro-step。多个数据并行 rank 完整均匀分片时，全局样本数 = 每 rank 微步数 × world_size。例如全局 batch 16 在 4 个 rank 上，每个 rank 处理 4 条，不能一律说“累积 16 次”。

源训练器不是把每条序列的平均 loss 再除以 16。它先数出本次更新所有可训练 token 的总数 `global_tokens`，再让每条序列的 **NLL 总和 / global_tokens** 反向。因此，一个更新步中的每个被监督 token 拥有相同权重，长度不同的轨迹不会被误当作相同数量的 token。

例如一条轨迹有 2 个监督 token，另一条有 6 个，总数为 8。先各自取均值再平均，会让短轨迹每个 token 的权重为长轨迹的 3 倍；先求和再除以 8，才是此处的 token-level mean。这是构造例子，不是项目数据分布。

![一次更新中的微步与统一归一化（示意，不是测量）](images/ch01_training_step.svg)

配套程序比较“大批一次反向”和“按同一总 token 数分母累积反向”的参数更新，并加入不同长度样本的断言。等价性依赖条件：参数在 micro-step 之间不能提前更新，归一化必须相同；dropout、跨样本运算和浮点求和顺序可能带来差异。不要把玩具例子的数值一致提升为任何模型都逐位一致。

最后才做梯度裁剪与 `optimizer.step()`，然后清梯度。源项目的顺序是：累积 → 跨 rank 梯度求和 → `clip_grad_norm_` → `optimizer.step()` → scheduler → 清梯度。若每条轨迹都先裁剪，等效优化问题就变了。来源：`train/sft_trainer.py::train`，提交 `ca57456`。

## 1.3 nn.Module、state_dict 与恢复训练

### 参数为什么要放进 Module

`nn.Module` 负责组织层、参数和子模块。把线性层赋给 `self.proj` 后，它会出现在 `parameters()` 与 `state_dict()` 中；临时放在普通 Python 容器里却没注册的层，可能不会按预期被优化器找到。多个子层可用 `ModuleList` 或 `Sequential` 注册。

```python
import torch
class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
    def forward(self, x):
        return self.proj(x)

model = TinyModel()
assert "proj.weight" in model.state_dict()
assert model(torch.ones(1, 3)).shape == (1, 2)
```

`state_dict()` 是参数和持久 buffer 的名字到张量的映射，不是完整的训练现场，也不会替你保存模型的 Python 类。恢复时仍要构造兼容的结构，再载入状态。若只是把返回的字典留在内存中作为“最佳模型”，应注意张量可能仍引用当前存储；需要独立快照时用深拷贝或立即写盘。

### “能加载权重”与“从这里接着训练”不是一回事

推理常常只需权重、配置、分词器和模板。恢复训练则还需要优化器动量、学习率调度状态、全局步数、数据位置与随机数状态。否则，同一份权重可能从不同学习率、不同样本或不同 dropout 开始，得到不同后续轨迹。

配套程序先走一步，再保存模型、优化器和 CPU 随机数状态；随后分别走“不间断第二步”和“新建对象后恢复第二步”，断言参数一致。示例在临时目录写入自己生成的 checkpoint，不加载陌生下载文件。对于真实工程，不应把不可信文件交给可执行任意对象反序列化的加载方式。

源项目使用 LoRA adapter 文件，而不是只把整个模型写成一个 `.pt`。`save_checkpoint` 还保存 `optimizer.pt`、`scheduler.pt`、训练元数据与各 rank 随机状态，最后写 `.complete` 标记。这里最后一步非常重要：目录存在不等于保存完成；中途被抢占可能只写下部分文件。恢复时还需校验配置与 manifest，防止拿着旧权重悄悄换了训练数据。

来源：`train/sft_trainer.py::save_checkpoint` 与恢复分支，提交 `ca57456`。本章例子只覆盖单进程 CPU 的最小闭环，不声称覆盖分布式、云端发布或故障原子性。

## 1.4 省显存的三件工具，各自解决不同的问题

### bf16：减少部分张量的存储精度

上一节已经比较了元素字节数。正式模型中还要审计哪些计算保留 fp32、优化器状态是什么类型，不能机械地把所有张量转换为 bf16。源配置的 bf16 与 fp32 输出头是明确的组合；教学程序演示类型转换及 CPU bf16 autocast 下的前向、fp32 loss 与反向，不测量性能收益。

### Gradient checkpointing：少留激活，反向时重算

这里的 checkpoint 不是磁盘断点。它指前向时不保留某些中间激活，反向需要时重算，用额外计算换存储空间。配套程序比较同一个小模块启用和未启用 checkpoint 的输出及梯度，而不声称测出了正式模型的显存节省比例。

```python
from torch.utils.checkpoint import checkpoint
# block 是已定义的模块，x 是模块输入
# non-reentrant 模式允许输入不要求梯度、内部参数需要梯度
# y = checkpoint(block, x, use_reentrant=False)
```

实际调用见配套脚本。重算必须与原前向语义一致：如果函数依赖可变全局状态、外部副作用或不同设备上的随机操作，要额外小心。源项目显式使用 `use_reentrant=False`，同时关闭 `use_cache`，并启用输入梯度支持。缓存推理 KV 和训练重算不是同一件事，不能不加检查地把推理设置照搬到训练。[PyTorch checkpoint 文档](https://docs.pytorch.org/docs/2.10/checkpoint.html)

### LoRA：冻结大矩阵，学习一个低秩增量

一个线性层的权重为 `W`，LoRA 用两个较小矩阵构造增量。以列向量记法写，输出为 `W x + (alpha / r) B A x`。`r` 是低秩维度；常规 LoRA 缩放为 `alpha/r`。如果输入维度为 `d_in`，输出为 `d_out`，新增参数量是 `r × (d_in + d_out)`，而全量矩阵有 `d_in × d_out` 个参数。这个公式没有计算 bias 或其他模块，也不保证任意小矩阵设置都能节省参数。

LoRA 是已有的公开方法，参见 Hu 等人的 [LoRA 原论文](https://arxiv.org/abs/2106.09685)。本书所说的 Apexo 是使用、组合与调整这些技术的项目流程，而不是重新命名 LoRA 的发明来源。

![冻结基座与可训练低秩旁路（示意）](images/ch01_lora.svg)

配套程序冻结一个小线性层，只更新 A、B；检查基座梯度为空、基座权重不变，而适配器能够更新。常见初始化是 A 随机、B 为零，因此初始增量为零；第一步 A 的梯度可能为零并不自动意味着程序坏了，要检查 B 的更新及后续步骤。

本项目配置使用 `r=64`、`alpha=64`、`dropout=0.0`，所以常规缩放因子为 1。适配器挂在以下模块：

- 全注意力投影：`q_proj`、`k_proj`、`v_proj`、`o_proj`。
- Gated DeltaNet 相关投影：`in_proj_qkv`、`in_proj_z`、`in_proj_a`、`in_proj_b`、`out_proj`。
- MLP：`gate_proj`、`up_proj`、`down_proj`。

这些是这个模型和配置的模块名，不能机械套用到任意模型。`train/model_utils.py::apply_lora` 使用 PEFT，并检查输出头仍然冻结。来源：该文件及 `configs/sft/qwen35_2b_lora.yaml`，提交 `ca57456`。CPU 教学使用较小的 r，目的是看清矩阵运算，不是重现实验配置。

三种技术可以组合，但互不替代：bf16 改精度，checkpointing 改激活保存策略，LoRA 改可训练参数集合。长序列、词表 logits、临时副本和推理 KV 仍可能主导峰值。

## 1.5 交叉熵、移位与 assistant-only mask

### 模型在位置 t 预测位置 t+1

自回归语言模型不是拿当前位置的 logits 预测同一个已输入 token。若输入是 `[a, b, c, d]`，前面三个位置依次用于预测 `[b, c, d]`。所以手写 loss 常配对 `logits[:, :-1]` 与 `labels[:, 1:]`。如果模型 API 内部已经移位，外面再移一次会错；先检查接口，再写代码。

交叉熵接受未归一化的 logits 和整数目标类别。对目标 token，它对应负对数概率。不要先做 softmax 再把概率传给期待 logits 的 `cross_entropy`，否则相当于多做了一次不同的归一化。[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.10/generated/torch.nn.CrossEntropyLoss.html)

### 不监督用户的话，不等于删掉用户的话

SFT 中，system、user 和工具返回是上下文；模型需要看见它们，但不必把它们当作要生成的目标。把相应 label 写为 `-100` 是一种常见实现：输入仍在，损失跳过对应目标位置。它不是 attention mask。被忽略位置的输入仍会影响后续被监督 token 的预测，因此其上下文路径仍可能参与梯度传播。

```python
import torch
import torch.nn.functional as F
logits = torch.zeros(1, 4, 6, requires_grad=True)
labels = torch.tensor([[-100, -100, 2, 3]])
targets = labels[:, 1:]
count = (targets != -100).sum()
assert count.item() == 2
loss_sum = F.cross_entropy(
    logits[:, :-1].reshape(-1, 6), targets.reshape(-1),
    ignore_index=-100, reduction="sum",
)
loss = loss_sum / count
loss.backward()
```

本例中，位置 1 的预测对应 label 2，位置 2 的预测对应 label 3。监督掩码跟随目标移位，而不是随便切掉一头。若一个样本没有任何可训练 token，必须跳过或报错，不能除以零，也不能把“没有监督”假装成成功训练。

### 真实项目比“role 等于 assistant”更严格

源项目 `train/dataset.py` 并非给所有 assistant 文本打标签。默认只训练带正确生成前缀、且包含一个有效 bash 调用的轮次。生成前缀本身不算目标；前缀之后的推理文本、关闭思考标记、内容、工具调用以及结束标记才构成目标段。system、user、工具返回和无效轮次保持忽略。

还要注意分词边界：字符边界不是 token 边界。一个 token 可能跨越“前缀结束”与“生成内容开始”。源代码分段 tokenize 再拼接，使训练输入与 rollout 的增量拼接方式一致；超长样本直接丢弃而非截断。这些选择将在第 5 章详细验证，本章不把简化 mask 当成完整的数据处理器。

### 按 token 归一化，把第二节闭合起来

对一个更新步，设所有有效目标 token 的集合为 S，目标就是这些位置的 NLL 总和除以 `|S|`。源训练器对每条轨迹求和，再统一除以全局监督 token 数；多设备时分母跨 rank 求和，梯度也按项目约定求和。不要一边用全局分母，一边又在通信层额外平均，却忘了补偿缩放。

源代码避免一次展开整个 `[T, V]` fp32 logits，而在 `chunked_token_logprobs` 中分块处理，并结合重算。分块应改变资源使用方式，不应偷偷改变监督位置或归一化目标。配套程序验证忽略位置、移位、token 数与手算负对数概率之间的一致性。

来源：`train/dataset.py::build_sft_example`、`train/model_utils.py::sequence_logprobs`、`train/losses.py::chunked_token_logprobs`、`train/sft_trainer.py::train`，提交 `ca57456`。

## 1.6 Transformers、分词器和聊天模板

### 加载的是一组相互匹配的约定

`AutoTokenizer` 负责把文本与 token ID 联系起来；`AutoModelForCausalLM` 根据配置选择因果语言模型结构。名字里的 Auto 不代表“任何权重都能互换”。模型结构、词表大小、special token、revision 与聊天模板必须匹配。

教学程序在临时目录创建极小的 tokenizer 与随机初始化模型，保存后通过 Auto 接口从本地加载，并设置只读本地文件。这样可以练习真实 API，不必下载大模型。随机模型没有学会对话；即使它能输出 token，也不能用输出质量评估后训练。

源项目则固定官方模型 revision，加载 tokenizer 后安装项目模板；模型按指定 dtype 与 attention implementation 加载，关闭缓存，再按需开启重算。其导出还有官方权重布局与文本模型布局的名称映射。本章只解释“加载接口”的基础，不提供一个未经验证的通用 Qwen 导出捷径。

### Chat template 不是装饰字符串

用户看见的聊天是角色和消息组成的列表，模型看见的是一个 token 序列。模板决定角色边界、消息结束符、工具描述，以及生成从哪里开始。换模板可能改变模型输入，即使可见的自然语言句子没有变。[Hugging Face 聊天模板说明](https://huggingface.co/docs/transformers/chat_templating)

训练通常已经有 assistant 的完整答案；推理则常需模板追加 assistant 开始标记，即 `add_generation_prompt=True` 的用途。先渲染为文本再 tokenize 时，要避免重复添加模板已含的特殊 token。配套例子检查训练渲染与生成前缀的不同，并检查本地保存加载后模板不丢失。

正式项目的模板带有推理和工具调用规则。教学模板只用于解释角色串行化，不是 Qwen3.5 原生模板，不应复制去正式训练。第 5 章会把模板、增量拼接和监督 mask 放在一起检验，因为三者必须描述同一条序列。

来源：`train/model_utils.py::load_tokenizer`、`load_policy`，提交 `ca57456`。公开 API 背景见上方官方文档。

## 1.7 把概念放回真实训练器

| 本章概念 | 源文件（提交 `ca57456`） | 阅读时应回答的问题 |
| --- | --- | --- |
| dtype、冻结头、LoRA、模型加载 | `train/model_utils.py` | 哪些参数可训练？哪些副本是 fp32？ |
| 分块 log-probability | `train/losses.py` | 目标移位在哪完成？重算覆盖什么？ |
| 累积、裁剪、优化器、断点 | `train/sft_trainer.py` | 一步更新有几条序列？分母是什么？ |
| 标签与模板边界 | `train/dataset.py` | 哪些 token 仅作为上下文？ |
| 固定实验设置 | `configs/sft/qwen35_2b_lora.yaml` | 配置是否真的传到对应函数？ |

建议先读训练循环，再沿调用关系去看辅助函数。读完本章，你不需要背出所有模块名，但应该能指出五类严重问题：没有调用 backward、每条样本都错误清梯度、用不匹配的分母、label 与 logits 错位、只恢复权重却声称完整续训。

### 小练习

1. 把例子中两条序列的监督 token 数改成不同长度，比较“均值的均值”与“总和除总数”。它们何时相等？
2. 在累积循环中故意每次 `zero_grad()`，哪条等价性断言会失败？
3. 将 LoRA 基座误设为可训练，检查可训练参数集合，而不是仅看 loss 是否下降。
4. 保存权重但不恢复 AdamW 状态，比较第二步结果；解释“模型能加载”为什么不是充足验收。
5. 将 labels 少移位一次，手工标出每个 logits 位置正在预测谁。

参考答案：等长监督或恰巧相同均值时两种聚合可能重合；每次清零会丢掉之前微步梯度；基座是否冻结要看梯度、参数集合和更新前后权重；优化器状态会改变后续更新；因果预测应始终从当前位置指向下一个目标 token。

### 本章边界与待作者确认

本章教学代码是独立的局部演示，不复现 GPU 显存峰值，不启动真实命令沙箱，也不声称复现完整训练。项目参数来自固定提交，不据此推断作者当时的心理或成本。当前没有新增待作者确认的问题；如后续版本发现配置与执行日志不同，应保留两者并说明差异，而非悄悄改写事实。

## 本章出处

项目事实固定于源仓库提交 `ca57456`：

- `requirements.txt`：GPU 训练依赖；不与 CPU 教学依赖混淆。
- `configs/sft/qwen35_2b_lora.yaml`：bf16、LoRA r/alpha、目标模块、batch 等配置。
- `train/model_utils.py`：模型与 tokenizer 加载、冻结头、LoRA、分块接口。
- `train/losses.py`：分块 log-probability 与重算。
- `train/sft_trainer.py`：累积归一化、梯度裁剪、优化器及恢复。
- `train/dataset.py`：有效轮次、label、分段 tokenization 与超长丢弃。

公开技术出处见各节 PyTorch、Hugging Face 官方文档和 LoRA 原论文链接。教学代码、图形和人为构造数字均可在本书 `examples/ch01_pytorch.py`、`scripts/draw_ch01.py` 与对应测试中复算。
