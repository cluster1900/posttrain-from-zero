# 第 5 章 聊天模板与工具调用格式

一条任务已经被冻结，模型也能计算梯度，下一步是否只需把对话交给 tokenizer？还不够。对于终端智能体，角色边界、思考段、函数名、参数名、工具返回和停止符共同构成训练合同。相同的自然语言经过不同模板，可以变成不同的 token 序列；训练器学习一种格式、推理端生成另一种格式时，权重即使正确同步，行为概率也不再是在同一条件下计算的。

本章从项目的 `configs/chat_templates/qwen35_agent.jinja`、`train/chat_format.py` 与 `tests/test_chat_format.py` 出发。项目证据固定为提交 `ca57456`。本章不发布完整上游模板或提示词；图与小实验独立编写，外部 Qwen 模板及参考 harness 的归属见许可附录。

## 5.1 模板不是聊天界面的装饰

对人而言，“用户说了什么”和“工具返回了什么”容易区分；对自回归模型而言，输入最终仍是一串 token。聊天模板负责把有结构的消息列表编码成模型约定的文本边界。`role=user` 是消息对象里的字段，`<|im_start|>user` 则是渲染后的文本框架，两者不能混为一谈。

源项目把同一模板装到 tokenizer 上，用于 SFT 数据渲染、RL 初始提示和导出的 checkpoint。模板文件不是只在第一次下载模型时使用；导出和服务阶段也必须保持一致。这里要固定的是**合同和版本**，不只是“都叫 Qwen3.5”。原项目测试还检查官方权重布局与文本模型加载的对应关系，第 6 章再展开。

| 层次 | 本项目中的例子 | 需要核对什么 |
| --- | --- | --- |
| 消息对象 | system、user、assistant、tool 及工具调用结构 | 角色、顺序、内容是否相同 |
| 模板文本 | 角色边界、思考标签、XML 风格调用 | 换行、空白、停止符是否一致 |
| token ID | tokenizer 编码结果 | tokenizer revision、特殊 token 与编码边界 |
| 训练目标 | labels 与 mask | 只监督允许学习的生成区段，而非所有输入文本 |

这四层都可能出错。文本相等是重要检查，但不能自动证明“各片段分别编码后的 ID 拼接”等于“整段一次编码”；分词算法可能跨文本片段边界合并字符。实际工程还必须检查 token 追加与 logprob 对齐。本章 CPU 小例子只验证文本及 UTF-8 字节层，不冒充正式 tokenizer 回归。

技术背景可参阅 Hugging Face 的[聊天模板文档](https://huggingface.co/docs/transformers/main/en/chat_templating)。正文的项目行为仍以固定代码为准，不把随版本更新的网页默认值倒写进历史记录。

![图 5-1　同一对话的四层合同：消息、模板文本、token 序列与监督掩码。结构示意。](images/ch05_contract.svg)

## 5.2 一次 assistant 行动的边界

源 `GENERATION_PREFIX` 以 assistant 角色开始，并进入 `<think>` 段。模型续写的是这个前缀之后的内容，而不是重新生成一次 assistant 头。思考结束后，调用采用嵌套结构：外层 `tool_call`，内层 `function=bash`，参数名为 `parameter=command`。第 3 章 v1 的 `<command>` 写法不能仅凭看起来像 XML 就当作同一协议。

下面是本书人为构造的文本示意，不是从训练数据取出的轨迹，也不会执行其中的命令：

```text
<|im_start|>assistant
<think>
检查虚构目录中的文件名。
</think>

<tool_call>
<function=bash>
<parameter=command>
printf 'toy'
</parameter>
</function>
</tool_call><|im_end|>
```

这个例子里有三种不同的“结束”。`</think>` 结束思考段；`</tool_call>` 结束调用文本；`<|im_end|>` 结束 assistant 轮。它们不能互换。解析器识别调用后，环境还要核对工具名称。源项目在 RL 环境和 SFT 数据里对多个调用的处理不同：`train/agent_env.py` 按出现顺序执行同一轮里的每个调用，每个调用消耗一步回合预算，遇到提交即结束（提交 `a5a8046` 起，对齐参考实现）；SFT 的 `_valid_bash_calls` 则只把恰好一个非空 bash 调用的轮次当作可训练轮。

思考段里提到一个调用，不应触发工具。`parse_assistant_response` 先分离思考与正文，未闭合 `<think>` 的回答返回没有工具调用的状态；测试专门覆盖思考段中的伪调用。相反，一段没有显式思考标签的完整 assistant 内容也可能出现在评测轨迹中，源解析器对此有单独分支。因此不能把本章较严格的教学解析器当成源解析器的逐字替代。

## 5.3 工具观测如何进入下一轮

工具返回不是模型自己说的话。项目把一个或多个连续 tool 消息包装成一个 user 轮，其中按调用顺序放置多个 `tool_response` 块，随后再打开新的 assistant 生成前缀。这里的 user 是**模板表示层角色**，并不表示观测来自真人。

下面五行是项目静态源码摘录，逐字核对 `train/chat_format.py` L190–194，提交 `ca57456`。它位于 `observation_suffix` 函数内部，不作为独立程序执行；无敏感标识替换。

```{.python .source-excerpt}
    observations = [observation] if isinstance(observation, str) else list(observation)
    if not observations:
        raise ValueError("observation_suffix needs at least one observation")
    blocks = "".join(f"\n<tool_response>\n{o.strip()}\n</tool_response>" for o in observations)
    return f"\n{IM_START}user{blocks}{IM_END}\n{GENERATION_PREFIX}"
```

注意第一行兼容单个字符串和字符串列表；空列表被拒绝，而不是悄悄生成一轮空观测。`strip()` 是合同的一部分，不能一边保留观测边缘空白、一边删除空白，再要求两种路径相同。

格式化观测时，源代码将 stdout 与 stderr 合并，附上退出码；没有输出时使用明确的占位文本。过长输出保留头部 5,000 字符和尾部 5,000 字符，提示中间省略量，触发阈值为 10,000 字符。这里计量的是字符，不是 token。不能据此推算固定的模型上下文开销，也不能把被截断的工具输出当作完整文件内容。出处：`train/chat_format.py` 的常量及 `truncate_observation`、`format_bash_observation`，提交 `ca57456`。

## 5.4 增量拼接与整段重渲染必须相遇

多轮 rollout 为节省重复工作，会保留已生成的序列，再追加工具观测和下一轮前缀。SFT 数据准备更像把整份消息历史一次渲染出来。两种路径如果在空白或角色边界上不同，训练器重算的条件序列便可能偏离生成时看到的序列。

原项目测试构造多轮调用：普通输出、错误输出、多行命令、最终提交，以及多个连续工具观测。它比较增量文本与模板整段渲染。最后一轮完整渲染会补一个换行，因此测试明确比较 `full == inc + "\n"`。把这个换行差异忽略掉再声称“完全相同”，和把它误判为模型误差，都不合适。

![图 5-2　两条构造路径在同一文本边界比较；末尾换行和 token 追加另有明确合同。](images/ch05_incremental.svg)

读者可在第 1 章教学环境中运行：

```bash
python examples/ch05_chat_template.py
pytest -q tests/test_ch05_chat_template.py
```

例子使用虚构任务和独立编写的简化序列化器，不联网、不读取正式 tokenizer、不执行任何命令。测试包括多行命令、连续观测、思考中的伪调用、未闭合思考、多个调用、错误工具、尾随正文和分隔符注入。教学语法主动拒绝部分复杂内容；它是一个可理解的合同实验，不是可以部署的通用 XML 解析器或沙箱。

下面的代码与 `chapter_snippet` 完全相同。字符串“工具调用文本”只是教学占位，不是合法可执行调用；本段只观察轮次结束边界：

```python
prefix = "<|im_start|>assistant\n<think>\n"
emitted = "检查当前目录。\n</think>\n\n工具调用文本<|im_end|>"
assembled = prefix + emitted
assert assembled.endswith("<|im_end|>")
assert "<|endoftext|>" not in assembled
```

## 5.5 停止符不是可以事后补上的标点

历史记录包含两个不同问题，不能揉成一次修复。

第一类是 **rollout token 追加与训练掩码**：生成引擎实际吐出的 token、为下一轮构造的框架文本、应参与 loss 的 token 必须区分。源提交 `33f3cc5` 修复的是生成以外来 `<|endoftext|>`／eos 而非 `<|im_end|>` 终止的情况：保留已采样正文的训练资格和环境执行，把最后那个外来停止 ID 及其 logprob 从训练序列移除，再用一个不参与 loss 的合成 `<|im_end|>` 关闭该轮；同时计数这类轮次。若触发长度截断，另按截断规则屏蔽，而不是混用这条分支。这里的外来停止符原本可能确实由引擎返回，不能误说成全部是程序额外插入的。

第二类是 **推理停止配置**：风险 R48 记载官方 Base 的默认 eos 只含 `<|endoftext|>`，而项目的 assistant 轮按 `<|im_end|>` 结束。vLLM 若不把正确轮结束标记作为停止条件，就可能越过该边界继续生成。后续修复加入轮次停止符；这不意味着修改所有模型的全局 eos 就一定正确，也不意味着每个停止 token 都应被删掉或全部纳入训练。

因此需要分别问：引擎何时停止？返回的 token 是否包含停止符？追加新上下文时是否重复或漏掉框架？训练 mask 是否只覆盖本次真正生成且允许监督的部分？这四个问题不能用“看起来回答结束了”代替测试。R48 的实际长生成与评测后果在第 11 章按发生时间展开，本章只建立格式概念，不把后期事故倒填成最初 SFT 失败。

## 5.6 排错时保留最小反例

| 现象 | 优先检查 | 不能据此直接下的结论 |
| --- | --- | --- |
| 看到了 bash 字样却没有行动 | 调用是否在闭合思考后的正文，标签与参数名是否正确 | 不一定是模型能力低 |
| SFT 可运行、rollout 重算概率偏差大 | 同一模板、tokenizer、增量边界和权重版本 | 不应先靠放宽误差阈值解决 |
| 多个 tool 返回被渲染成多轮 user | 连续观测分组与末尾换行 | 角色名相同不代表字节相同 |
| 模型越过一轮边界继续生成 | eos／stop 配置及返回 token 保留规则 | 不一定是 GPU 内存不足 |
| 某次工具输出缺少关键内容 | 字符截断规则、退出码和原始产物 | 不能把可见片段当作完整输出 |

最有效的排错材料通常不是一张长截图，而是一份最小消息列表、两条构造路径的文本差异、对应 token ID 与 mask 位置。把这些保留下来，后续才有机会区分数据清洗问题、模板问题、模型同步问题和真实策略行为。

## 5.7 在 5,795 条真实轨迹上量一量边界

前面的教学例子只比较文本。这一节换成真实数据：源项目 SFT 实际读取的 5,795 条成功轨迹（数据 revision `9d6ab74`，parquet SHA-256 `7507b499…e38a5`，与 round1 交接记录一致），直接调用源项目 `train/dataset.py` 的 `build_sft_example`（提交 `ca57456`），用 Qwen3.5-2B-Base tokenizer（revision `b1485b2`）编码。脚本是 `examples/ch05_token_boundary_audit.py`，汇总写入 `data/ch05_token_boundary.json`；在 CPU 上全量跑一遍约 286 秒。【读者运行所得】

先核对这次复现和项目记录是否同一份数据：按 16,384／24,576／32,768 的长度上限，分别保留 5,330／5,618／5,711 条；全部样本共 48,640,561 个 token，其中 21,959,803 个参与 loss。这四个数与源 `docs/apexo/03_data_plan.md` §3.3 记录的 5,330／5,618／5,711、48.6M、21.96M 一致，所以下面的统计说的就是项目训练用的那份数据。

![图 5-3　5,795 条成功轨迹的 assistant 轮数分布：中位数 11 轮，最少 3 轮；64 轮处有 47 条堆积，而 50 轮以上其他各档都不超过 7 条，推测采集时有 64 轮上限（数据里没有记录上限本身）。数据：data/ch05_token_boundary.json。](images/plots/ch05_turns.svg)

轮数决定了边界的数量。每个 assistant 轮在渲染文本里切出两个边界（生成前缀之后、`<|im_end|>` 之后），全数据共 164,406 个边界。项目的做法是**分段编码**：在每个边界处把文本切开，各段单独编码再拼接（`train/dataset.py` L7–10 的注释写明了理由）。如果改成整段一次编码，会有多大差别？

| 统计项 | 数值 |
| --- | --- |
| 分段编码与整段编码 token 序列完全相同的轨迹 | 754 条 |
| 两者不同的轨迹 | 5,041 条（87.0%） |
| 被整段编码的某一个 token 跨越的边界 | 24,977 / 164,406 |
| 其中位于 assistant 片段起点的 | 24,977（全部） |
| 分段编码多出的 token 总数 | 24,977（占 48,640,561 的 0.051%） |

每条轨迹“分段比整段多出的 token 数”都恰好等于它被跨越的边界数，5,795 条逐条相等。被跨越的边界全部是同一种：思考为空的 assistant 轮，渲染出 `<think>\n\n</think>`，而切口正好落在两个换行之间。最小例子如下（ID 来自 `data/ch05_token_boundary.json` 的 `minimal_example`）：

| 编码方式 | token |
| --- | --- |
| 分段：前缀 | `<|im_start|>`(248045) `assistant` `\n` `<think>`(248068) `\n`(198) |
| 分段：生成部分 | `\n`(198) `</think>`(248069) `\n\n`(271) `<tool_call>` |
| 整段一次编码 | … `<think>`(248068) `\n\n`(271) `</think>`(248069) … |

整段编码把两个换行合成了 ID 271，这个 token 一半属于提示、一半属于模型生成。rollout 时模型拿到的提示以 `<think>\n` 结尾，接下来生成的第一个 token 只能是 198，不可能是 271。所以分段编码才和推理时的 token 序列一致；整段编码会让 SFT 在 30.4% 的 assistant 轮（24,977 / 82,203）上学一个推理时永远不会出现的开头，监督掩码也无法干净地落在这个 token 上。

![图 5-4　每条轨迹分段编码比整段编码多出的 token 数分布；0 表示两者相同（754 条）。多出的 token 数等于被跨越的边界数。数据：data/ch05_token_boundary.json。](images/plots/ch05_token_boundary.svg)

同一次运行还给出了数据清洗计数（`build_sft_example` 的 stats）：528 条 assistant 消息的正文里混有思考标签，标签被删掉；1,333 个轮次渲染时没有 `<think>` 前缀，不参与训练；378 个轮次没有工具调用，2 个轮次的工具调用不合规，也都不参与训练。5,795 条全部至少保留一个可训练轮，没有因此整条丢弃。

## 坑与修复

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 空思考轮整段编码后 `\n\n` 合成 ID 271，横跨提示与生成边界（全数据 24,977 处） | BPE 跨字符串边界合并；推理时提示以 `<think>\n` 结尾，模型先生成 198 | 在每个片段边界分段编码，再拼接 | `train/dataset.py` L7–10，引入于提交 `fbf621e`（10-11）；本章 5.7 的全量复核 |
| assistant 正文里混有 `<think>`／`</think>` 标签（528 条消息） | 原始轨迹把思考标签写进了 content，渲染后思考段边界会错位 | 删除标签并计数 `stray_think_tag_removed`（528 处） | `train/dataset.py` L72，提交 `fbf621e` |
| 格式错误反馈之后，前面的空思考轮渲染时没有 `<think>` 前缀 | 源注释原话：format-error feedback user turns make earlier empty-reasoning turns render without `<think>` | 无前缀的轮次不打标签，计数 `turn_without_think_prefix`（1,333 个） | `train/dataset.py` L14、L129，提交 `fbf621e` |
| 一轮里出现多个工具调用时，执行方式与参考实现不一致 | 环境层的多调用处理没有对齐参考实现的工具循环 | 按出现顺序执行每个调用，每个调用消耗一步预算，遇到提交即结束，每个调用返回一个观测 | `train/agent_env.py`（多调用循环），提交 `a5a8046`（10-12） |
| 生成以 `<|endoftext|>` 而不是 `<|im_end|>` 结束，外来停止符进入训练序列 | 官方 Base 的 eos 只有 248044 | 去掉外来停止符及其 logprob，用不计 loss 的 `<|im_end|>` 收尾并计数 | 提交 `33f3cc5`（10-12） |
| 服务端越过轮次边界继续生成，16K 截断引发 `format_error` 和 HTTP 400 | `vllm serve` 没把 `<|im_end|>`(248046) 设为停止符 | `--override-generation-config` 设 `eos_token_id=[248046,248044]`，导出时写入 `generation_config.json` | R48；提交 `c33ee0d`（10-19） |

## 时间线、花费与教训

模板与分词合同的工作集中在 10-10 至 10-12：初始框架 `047c932`（10-10），分段编码与清洗计数 `fbf621e`（10-11），多调用顺序执行 `a5a8046` 和外来停止符修复 `33f3cc5`（10-12）。停止符在服务端的后果直到 round10 评测才暴露，修复在 `c33ee0d`（10-19）。这部分全部在 CPU 上开发和测试，没有单独的 GPU 花费记录；本节复核也只用了约 286 秒 CPU。

教训：

1. **在真实数据上量边界，不要只在玩具样例上比文本。** 文本完全相同的情况下，87% 的轨迹 token 序列仍然不同。
2. **以推理时的 token 序列为准。** 训练序列应该是“提示的编码 + 模型实际生成的 token”，而不是把整段文本重新编码。
3. **清洗要计数。** 528、1,333、378 这类数字让人能判断清洗规则到底影响了多少数据。

## 本章出处

- `configs/chat_templates/qwen35_agent.jinja`：项目统一模板及两项相对官方模板的改动说明。提交 `ca57456`。
- `train/chat_format.py` L190–194 及解析／观测函数；`tests/test_chat_format.py`：前缀、连续观测、增量／整段渲染、解析边界。提交 `ca57456`。
- `train/model_utils.py` 的 `load_tokenizer`、`train/dataset.py`、`train/rollout.py`：模板安装、监督与追加路径。提交 `ca57456`；停止符相关历史修复另见提交 `33f3cc5`。
- `docs/apexo/06_risks_and_decisions.md` R48；`docs/apexo/00_progress.md`：后期停止符问题的记录，未倒写成早期训练结果。提交 `ca57456`。
- 本书 `examples/ch05_chat_template.py` 与对应测试：原创文本级教学。
- 本书 `examples/ch05_token_boundary_audit.py`、`data/ch05_token_boundary.json`、`scripts/figures/ch05_07_real_data.py`：5.7 节的全量复核，调用源项目 `train/dataset.py`（提交 `ca57456`）；`docs/apexo/03_data_plan.md` §3.3 的保留条数与 token 总数用于交叉核对。提交 `fbf621e`、`a5a8046`、`33f3cc5`、`c33ee0d` 见坑表。
