# 第 4 章 数据：把 1.5 万个任务冻结下来

本章的“冻结”不是把一份数据下载到目录里就结束，而是把**下载了哪个版本、留下哪些任务、每份文件究竟是什么字节**固定下来。只有这样，后面谈训练变化时，才不会把数据变化误当成模型变化。

round1 在 2026-10-12 的 Colab CPU 会话完成。日志从 `05:11:42Z` 到 `05:31:45Z`，相减为 1,203 秒，即 20 分 3 秒。这是日志覆盖的数据准备与交接命令区间，不包含未记录的会话创建、人工等待或账户操作。源材料：`result/round1/logs/prepare_all.log`、`result/round1/README.md`，提交 `ca57456`。

## 动机

v1 把评测任务 registry 名当成了可供训练的数据集入口；v2 改为直接使用已经公开的终端任务包与成功 SFT 轨迹，不再在本项目中造一个任务合成器。这不是“自己生成的数据质量不好”的实验结论，而是源数据方案写明的范围选择。来源：`docs/apexo/03_data_plan.md` §1–2，提交 `ca57456`。

两份输入做不同的事。RL 任务包提供目标、环境镜像和验证材料，让策略自己尝试。SFT 轨迹则提供已经发生过的交互示范，让策略先学习工具使用格式与行动序列。它们不是同一份表简单改列名：一个任务可能对应多条轨迹，一条轨迹又包含多个 assistant 轮。因此本章会严格区分任务数、轨迹数、轮数和 token 数。

还要防止训练与留出任务泄漏。虽然源记录核验这两套任务 ID 没有重叠，准备脚本仍按 RL held-out 清单剔除 SFT 任务，校验器仍检查交集。当前交集为零是一条数据事实，不是删除防护的理由；以后改了 revision，事实可能改变。来源：`docs/apexo/03_data_plan.md` §3.3、`docs/apexo/06_risks_and_decisions.md` R16。

![图 4-1　冻结流程示意：先校验材料，再确定划分，再写 manifest 与哈希；无效输入不得留下像正式产物的文件。](images/ch04_freeze.svg)

## 配置

### 4.1 固定输入与三个输出

下表来自 round1 结果摘要与准备日志；具体公开数据集 ID、作者与许可证见致谢附录。本章不改名重发外部数据，也不把整份任务包复制进书仓库。

| 项目 | 固定值或结果 | 证据 |
| --- | --- | --- |
| 正式运行代码 | `ffd4713c8ba57057641d765afdc2b80a1dd1a44a` | `result/round1/README.md` |
| RL 源 revision | `7b090eca98bf351356bc1c64290c5c4a09f2f98c` | `result/round1/results/manifest_info.json` |
| SFT 源 revision | `9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4` | `result/round1/logs/prepare_all.log` |
| 划分种子 | 42 | 同上与 `manifest_info.json` |
| RL 全量记录 | 14,601 | `manifest_info.json` |
| RL 训练 / held-out | 14,401 / 200 | 同上 |
| 不同环境镜像 | 14,490 | 同上 |
| SFT 成功轨迹 / 不同任务 | 5,795 / 1,136 | `prepare_all.log` |
| SFT 与 RL 任务交集 | 0 | `docs/apexo/03_data_plan.md` §3.3 |

这里“1.5 万”是公开任务包的概称，精确输入行数是 14,601，不应在统计表里把它改写成 15,000。14,490 是镜像种类，不是这轮启动的容器数；round1 只是数据准备，没有因此运行一万多个沙箱。5,795 是轨迹数，不能与 1,136 个 SFT 任务混用。正文读取的是上述文件在 `ca57456` 的归档，实际执行代码版本另列，避免把后来的命名整理当成当时重新跑过数据。

归档中的三份 SHA-256 如下。SFT 行采用类别称呼，避免把历史文件名的旧标识继续带入正文；哈希未改写，读者可据此核对交接记录。

| manifest 类别 | round1 记录的 SHA-256 |
| --- | --- |
| RL train | `f95418e7240fa25f922e7149d7b5cadff82e37020193e107ebb7df37b1dcb4c0` |
| RL held-out | `4f03fc7ad7355311e6ba06857da7b2846b125e3676c9942b2400b763f72e9a37` |
| SFT 成功轨迹 | `879d349c827387a643e9503ccf3875a47e296e0be078d537084c074500c9437b` |

来源：`result/round1/results/data_handoff.json`、`result/round1/logs/prepare_all.log`，提交 `ca57456`。长哈希是数据指纹，不是访问密钥。

### 4.2 manifest 保存什么

RL manifest 的每行把 `task_id`、原始 `instruction`、镜像、相对任务目录、来源与 split 联系起来。只保存任务 ID 不够：模型收到的指令可能与任务包不同；只保存镜像也不够：同一个镜像可能被不同任务引用。源准备脚本逐条核对 `instruction.md` 渲染出的用户消息与 parquet 中的用户消息。round1 记录的 missing instruction、missing test.sh 和 instance mismatch 都为 0。

系统提示差异另算。日志显示 14,601 条源系统提示与项目覆盖提示不同，脚本给出警告，但没有把用户任务说明不一致也降成警告。前者是明确选择项目提示词，后者会改变任务内容。把所有 warning 都解释成失败，或把所有差异都当无害，都会读错这份日志。来源：`result/round1/results/manifest_info.json`、`scripts/data/prepare_apexo_rl.py`，提交 `ca57456`。

### 4.3 划分不依赖输入的到达顺序

源实现先计算 `sha256(f"{seed}:{task_id}")`，按摘要排序，前 200 个作 held-out，其余作训练。它不是每次重新调用一个没有固定状态的随机打乱函数。相同任务 ID、相同 seed 和同一套逻辑给出相同归属；输入文件行顺序变化不会改变归属。`--limit` 是调试开关，只在完整校验通过后、划分之前截取；全量运行不使用它。

以下是源代码原样摘录，只展示排序与划分调用，不执行，也不包含下载端点。来源：`scripts/data/prepare_apexo_rl.py` L55–62，提交 `ca57456`。

```{.python .source-excerpt}
def select_and_split(records: list[dict], *, seed: int, heldout_size: int, limit: int = 0):
    """Sort by the deterministic split key, optionally keep the first ``limit`` tasks, then split."""
    if type(limit) is not int or limit < 0:
        raise ValueError(f"--limit must be an integer >= 0, got {limit!r}")
    ordered = sorted(records, key=lambda r: split_key(seed, r["task_id"]))
    if limit:
        ordered = ordered[:limit]
    return split_records(ordered, heldout_size)
```

排序只是一个环节。源 `split_records` 还拒绝让 held-out 大小大于等于剩余任务数，因为这样会写出空训练集。各 split 最终按 task_id 排序后再写文件，便于稳定序列化。任务归属稳定与文件字节稳定是两个层次；行顺序、JSON 格式或换行变了，即使语义相同，SHA-256 也会变。

### 4.4 成功轨迹里仍然存在不应训练的轮

“only_success”只描述轨迹的最终结果，不保证每一个中间回复都适合模仿。源数据方案记录了四类清洗/标注问题：528 处残留思考标记、1,333 个缺思考前缀的 assistant 轮、378 个没有工具调用的轮，以及 2 条含非 bash 工具调用的样本。它们的计量单位不同，不能相加得到一个“坏样本总数”。

残留标记在 content/reasoning 规范化时删除；JSON 字符串形式的工具参数转为对象。默认不训练缺思考前缀、没有工具调用、非 bash 或参数非法的轮。这里“不训练某一轮”与“删除整条轨迹”也不同：一条轨迹可以留下有效的其他轮。如果清洗后没有可训练轮，样本才会被丢弃。来源：`docs/apexo/03_data_plan.md` §3.3、`train/dataset.py`，提交 `ca57456`。

round1 的 D2 校验报告明确列出 `timeout` 与 `cat` 两条非 bash 工具名警告，随后仍为 PASS。这不表示训练器会学习这两个工具名，而是校验器允许把该轮交给训练阶段 mask 掉。冻结的 5,795 条轨迹不应据此悄悄改写成 5,793 条。数据准备通过，和所有 token 都被监督，是两件事。

![图 4-2　三个层次不要混淆：任务决定划分，轨迹承载交互，assistant 轮决定监督区段。示意图不表示实际样本比例。](images/ch04_units.svg)

### 4.5 长度阈值为什么是整条丢弃

源记录的项目模板/分词器统计为：p50 6,009、p90 14,586、p95 20,367、p99 36,454，最大值写作 345K；本书保留这一显示精度，不给 345K 编造一个精确整数。统计还记载总计 21.96M 标注 token、48.6M token。它们是源文档报告值，不是本书重新取得全部数据后算出的结果。

`max_seq_len` 取 16,384 / 24,576 / 32,768 时，分别保留 5,330 / 5,618 / 5,711 条。2B 的 24,576 配置对应丢弃 177 条。源设计选择超长时整条丢弃，不把长轨迹裁掉尾部，因为尾部可能包含提交步骤。这个做法损失部分数据覆盖，但避免把成功轨迹切成没有最终动作的前缀。4B 后续 SFT 配置是 20,480；本表没有对应的精确保留条数，所以不从三个点插值冒充实测。

![图 4-3　源记录的 SFT 长度阈值与保留轨迹数；三个离散报告点，不插值。数据来源：docs/apexo/03_data_plan.md §3.3，提交 ca57456。](images/plots/ch04_retention.png)

图 4-3 由本书 `scripts/figures/ch04_retention.py` 从 `data/ch04_retention.json` 生成。保留总数与最终进入某轮训练的样本数不是永久等号：源实现还要判断可训练轮、模板与实际配置，因此后续训练章节应读取那轮日志，而不是只从本图猜算。

## 命令

### 4.6 实际入口与脱敏后的重建示意

round1 归档记录实际执行入口是 `bash scripts/data/prepare_all.sh`，成功后执行 `scripts/data/handoff.py create`。当时的代码版本和源 revision 已在上表列出。旧日志含历史命名的环境变量、目录与产物名；为避免公开私有目录及旧标识，下列命令按 `ca57456` 的对应入口作**规范化展示**，不是逐字符的历史 shell 记录。`/path/to/project-data` 是读者自定的占位路径；此命令会联网下载完整数据，**不是本书 CI 或本次写作实际执行的命令**。

```{.bash .source-excerpt}
# 规范化展示；需源项目环境与完整数据，不在本书 CI 执行。
export APEXO_ROOT=/path/to/project-data
SEED=42 HELDOUT=200 bash scripts/data/prepare_all.sh
python scripts/data/handoff.py create --data-root "$APEXO_ROOT/data"
```

流程不只下载文件。它依次准备 RL、按 held-out 清单准备成功 SFT 轨迹，然后运行 RL train、RL held-out 与 SFT 三次校验；最后交接记录包含三份 manifest 的哈希、源 revision、seed、held-out 大小与代码 commit。可选的 `SFT_ALL=1` 会生成含未成功轨迹的对照 manifest，但它不属于这轮三份正式交接产物，不能无说明替换生产 SFT 数据。来源：`scripts/data/prepare_all.sh`、`scripts/data/handoff.py`、风险 R33，提交 `ca57456`。

### 4.7 本书真正可运行的 CPU 小实验

安装方法沿用第 1 章。运行 `python examples/ch04_data_freeze.py` 与 `python -m pytest -q tests/test_ch04_data_freeze.py`。脚本只生成几条人工任务，在临时目录写入小文件，不联网、不启动容器、不读取项目凭据。

先观察下面的最小划分，代码与脚本 `chapter_snippet` 一致。四个 `toy-*` ID 是人为构造，不是原项目任务名。

```python
import hashlib
task_ids = ['toy-a', 'toy-b', 'toy-c', 'toy-d']
ordered = sorted(task_ids, key=lambda t: hashlib.sha256(f'42:{t}'.encode()).hexdigest())
heldout, train = set(ordered[:1]), set(ordered[1:])
assert not heldout & train
assert heldout | train == set(task_ids)
```

完整教学脚本还检查三个不变量：输入顺序反转后归属与输出字节不变；给输出多加一个空白字节后哈希不再相等；把一条任务的测试材料标成缺失后，准备失败并移除旧 manifest，不能让上一次成功产物伪装成本次成功。这个小例子只接受 `task_id`、`instruction`、`has_test` 三个字段，并明确检查类型与必需值；缺 ID 等无效输入也必须走同一清理路径。它演示输入校验失败的处理，不声称实现了文件系统故障下的多文件事务。教学序列化格式是为本书新写的，**不声称其哈希等于源项目 manifest 哈希**。

哈希只能帮助判断字节一致，不能证明数据内容正确，也不能替代可信来源。一个错误文件同样有稳定哈希。正确顺序是先验证内容合同，再记录完整文件的摘要，交接时同时核对合同与字节。技术背景参见 [Python hashlib 文档](https://docs.python.org/3/library/hashlib.html)。

## 时间线与耗时

| 事件 | 开始（UTC；SGT） | 结束（UTC；SGT） | 时长 / 状态 | 证据 |
| --- | --- | --- | --- | --- |
| 数据准备入口与交接创建 | 2026-10-12 05:11:42；SGT 13:11:42 | 2026-10-12 05:31:45；SGT 13:31:45 | 1,203 秒，20 分 3 秒，日志以 PREPARE_ALL_OK 结束 | `result/round1/logs/prepare_all.log` |
| 三份 manifest 校验 | 位于上述区间内；未逐项记录起点 | 位于上述区间内；未逐项记录终点 | RL train、held-out、SFT 都 PASS；不拆分下载/解包/校验耗时 | 同上 |
| Drive 刷新、卸载与会话停止 | 未记录独立时间戳 | 未记录独立时间戳 | 归档记为完成；不并入精确命令时长 | `result/round1/README.md`、`docs/apexo/00_progress.md` §6.20 |

SGT 为 UTC+08:00；本表两个精确时刻均在 2026-10-12。以上证据均为 `ca57456`。日志给的是命令级区间，不足以计算每个步骤占比，因此本章不画下载或 CPU 利用率时间曲线，也不从“约 20 分钟”反推出带小数的分阶段耗时。

## 坑与修复

下表同时列出实际警告、此前审核修正的缺陷和当时登记的交接风险；“风险”不写成 round1 必然已经发生的故障。后续 GPU 挂载失效、adapter 损坏与训练 checkpoint 的问题留给对应运行章节，不倒填成本轮事件。

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| v1 数据入口取错对象 | 把任务 registry 当成 HF 训练数据入口 | v2 使用公开任务包与 SFT 轨迹，固定 revision | `docs/apexo/03_data_plan.md` §1；`00_progress.md` §1.2 |
| 缺文件或任务正文不一致时仍可能留下缩水 manifest（审核发现） | 验证未失败关闭，旧产物可能被误认作本次结果 | 校验失败不写两份 RL manifest，并删除旧文件；旁注记 failed_validation | `00_progress.md` §6.15；`scripts/data/prepare_apexo_rl.py` |
| 调试 limit 后训练集为空（审核发现） | held-out 大小没有小于剩余任务数 | 全量先校验；limit 后检查划分大小，非法即失败 | 同上 |
| 解包中断后目录存在但材料不全（风险） | 把“目录存在”当作“解包完成” | 临时目录解包，完成标记绑定 revision 与 tarball 哈希，再改名 | `03_data_plan.md` §3.2；`prepare_apexo_rl.py` |
| 外来 tar 成员可能越过目标目录（风险） | 未限制路径、链接或特殊文件 | 使用 tarfile 的 data filter；另核验必需材料，不能把过滤器当作所有风险的证明 | `prepare_apexo_rl.py` L65–67 |
| round1 全部源系统提示与项目提示不同 | 项目有自己的提示覆盖 | 明确 warning，用户任务内容一致性仍必须通过；不静默替换任务正文 | `result/round1/results/manifest_info.json` |
| SFT 带残留思考标记、字符串参数 | 外部轨迹格式需要规范化 | 清理残留标记，将参数解析成对象 | `03_data_plan.md` §3.3；`train/dataset.py` |
| 缺思考前缀、无工具调用或非 bash 轮 | 并非每个 assistant 轮都符合训练合同 | 轮级 mask；两条非 bash 警告不改写冻结轨迹数 | 同上；`result/round1/logs/prepare_all.log` |
| 超长轨迹截断会丢提交步 | 末尾可能承载最终动作 | 超过长度上限整条丢弃，保留数另记录 | `03_data_plan.md` §3.3；`09_custom_training_design.md` §3.4 |
| 训练/held-out 任务可能泄漏（R16） | 数据版本或任务来源变化后不能假设永远不重叠 | 按 held-out ID 剔除 SFT，校验交集；本版记录交集为 0 | `06_risks_and_decisions.md` R16 |
| Drive 显示文件已写但远端仍可能不完整（R23/R27） | FUSE 异步回写、大量小文件与读写限额 | 刷新卸载后读回校验；默认按固定 revision 重建再核对哈希，不能只看交接 JSON 存在 | `06_risks_and_decisions.md` R23/R27；`03_data_plan.md` §3.4 |
| 用 all 对照替换成功轨迹生产数据（R33） | all 包含失败轨迹且不在三份正式交接中 | 作为显式对照单独处理，不默默晋级正式数据 | `06_risks_and_decisions.md` R33 |
| 几乎每任务一个镜像（R2） | 14,601 任务对应 14,490 镜像 | 数据阶段只冻结引用；实际拉取/限速与容量风险由 RL 环境阶段处理 | `manifest_info.json`；`06_risks_and_decisions.md` R2 |

表内源文件均为提交 `ca57456`，短路径位于 `docs/apexo/` 或 `scripts/data/`。Python 的 data filter 能拒绝部分危险成员，但不能排除所有解包资源耗尽或数据内容风险；参见 [Python tarfile 提取过滤器](https://docs.python.org/3/library/tarfile.html#extraction-filters)。本书不下载或解包外部任务，也不运行其中命令。

## 花费

这轮是 Colab CPU 数据准备，不是 GPU 训练。归档没有给出 CPU 型号、该会话单独账单或细分费用，不能拿 20 分 3 秒乘一个猜测的单价。作者 2026-10-19 补充的约 200 美元是整个项目的 Google Cloud 费用，Colab 计算单元不在其中；本章不把总额分摊到 round1，也不把“没有金额记录”写成“免费”。来源：`result/round1/README.md`、`docs/apexo/11_timeline.md`“作者补充”，提交 `ca57456`。

磁盘同样区分记录精度。数据方案写解包后约 210MB、约 4.4 万个小文件；这是存储规模描述，不是这轮网络流量、云盘账单或峰值内存实测。大文件加小文件树的差别解释了为何项目默认选择固定版本重建，而不是把所有小文件经 Drive API 再搬一次。来源：`docs/apexo/03_data_plan.md` §3.4、§6。

![图 4-4　交接的三类证据与接收端读回校验。概念示意，不是网络传输耗时。](images/ch04_handoff.svg)

## 教训

1. **固定 revision、固定划分、固定字节是三个互补约束。** 源 revision 回答输入版本；seed 与算法回答任务归属；manifest 哈希回答实际消费的文件。少一个都可能使“同一数据”变得含糊。
2. **失败应留下失败证据，而不是留下看起来正常的旧产物。** §6.15 的修复说明仅打印错误不够；旁注状态与旧 manifest 的处理也必须一致。教学脚本用缺测试文件的负例验证这一点。
3. **成功轨迹仍需要轮级标注。** 最终成功不代表每轮格式正确。不要把轨迹级筛选当成 token 级监督的替代品，也不要把轮数、样本数和任务数加在一起。
4. **交接完成要以读回为准。** round1 留下了哈希与完成记录；后续使用者仍需校验实际文件，不能把 Drive 路径存在或 JSON 文件存在当成整套数据已经可靠落盘。
5. **离线数据检查不等于环境可用。** D1/D2 PASS 不能证明镜像可拉取、Docker 可运行或 verifier 不会异常。下一阶段必须另验这些链路，而不是让模型训练替基础设施做探针。

## 本章出处

源仓库提交 `ca57456`，实际 round1 运行代码另见 `ffd4713c8ba57057641d765afdc2b80a1dd1a44a`：

- `docs/apexo/03_data_plan.md` §1–6：来源、划分、清洗、长度统计、交接与存储规模。
- `docs/apexo/00_progress.md` §2、§6.12、§6.15、§6.17、§6.20：冻结值、准备前审核、对照边界与实际完成记录。
- `docs/apexo/06_risks_and_decisions.md` R2、R16、R23、R27、R33：与本章有关的风险；并非每项都在 round1 实际触发。
- `docs/apexo/09_custom_training_design.md` §3.4、`train/dataset.py`：轨迹规范化、监督区段与长度过滤。
- `docs/apexo/11_timeline.md`、`result/round1/README.md`、`result/round1/logs/prepare_all.log`：日期、命令入口、起止时刻与结果。
- `result/round1/results/manifest_info.json`、`result/round1/results/data_handoff.json`：RL 计数、源哈希与三份交接记录。
- `scripts/data/prepare_all.sh`、`prepare_apexo_rl.py`、`prepare_apexo_sft.py`、`verify_manifest.py`、`handoff.py`；`tests/test_dataset.py`、`tests/test_manifest_and_eval.py`：实现与原项目测试。

公开数据的原名、固定 revision 和 ODC-BY 1.0 署名见致谢附录。书中的人工任务不来自这些数据集；本批只发布脱敏汇总数，不发布任务包或轨迹内容。
