# 第 8 章 算力去哪里找：Colab、GCP 与 Agent Platform（10-11 至 10-14）

## 动机

训练脚本能跑，并并非训练系统已经有地方跑。第 7 章的 Colab SFT 交出了 adapter；接下来还要在同一条流程里加载模型、启动推理引擎、运行不可信终端命令、保存 checkpoint，并在失败后确认没有遗留计费资源。GPU、容器权限、持久存储和控制面配额是四个不同问题。

项目最初曾把沙箱交给独立服务。作者在 10-19 的回顾中说明：已有 GCP VM 能承接这项工作，因此不再保留 Daytona 路线。这是有记录的选择，不是本书事后编出的性能或价格比较。10-11 的实现把 Colab SFT 与 GCP 训练/评测拆开；后续又从直接 VM 方案转向 GPU CustomJob 加临时 CPU VM。本章沿着实际试验写，不把未创建的机器当成成功运行。

这里的 Agent Platform 是项目中的提交、运行、回收与核验流程，不表示某个托管产品自动提供全部隔离保证。主要依据为 `result/round3/README.md` 至 `result/round9/README.md`、平台计划与进度台账，结项提交 `ca57456`。

![图 8-1 平台路线按已运行与未运行区分，箭头表示方案演进而非一次作业内部流程](images/ch08_routes.svg)

## 配置

### 8.1 先问机器必须提供什么，再问在哪里租

Colab SFT 需要模型加载、反向传播和可恢复保存；真实 RL 还需要多轮模型生成与原始任务环境。GPU worker 不一定允许启动嵌套容器，能够执行一个 shell 也并非能运行数据集规定的镜像。把这些要求列成合同，才不会把“成功创建作业”误判为“训练环境齐备”。

| 层 | 要验证的能力 | 不能代替它的信号 |
| --- | --- | --- |
| 计算 | GPU 型号、实际可见容量、dtype、有限梯度 | 配置文件写着 A100 |
| 运行时 | 固定依赖导入、目标模型、实际长度、同步 | 小型 Linear 更新成功 |
| 环境 | 原任务镜像、输入文件、执行器、测试与 reward 文件 | 托管 shell 能输出文字 |
| 持久化 | 完整 checkpoint、版本与哈希读回 | 上传函数返回成功 |
| 回收 | 所有临时资源逐项不存在 | 删除请求发出或控制器退出 |

round3 的直接 GCP VM 方案准备了 80GB 机型与启动步骤，但 README 明确写着未开始、没有创建 VM；后续平台方案取代了它。round4 才实际用单张 A100 40GB Spot CustomJob 做能力探针。它验证了三种精度的小矩阵反传和微型模型的本地保存，却没有安装完整模型训练栈，因此不能被写成 Qwen 正式训练成功。

### 8.2 40GB 与 80GB：先统一计量口径

早期方案考虑 80GB，Colab 正式 SFT 也确实使用过 80GB。这不自动证明 40GB 不够，更证明不了 40GB 足够。进度台账 §6.32 记录短链路在 40GB 上完成更新，但旧门禁把逻辑 allocated 统计和整卡物理占用混在一起。缓存与不同运行阶段的统计口径必须先拆开。

最终路线使用一张完整可见容量至少 39GiB 的 A100 40GB，训练与推理串行交替。源配置把 vLLM `gpu_memory_utilization` 初值设为 0.55、`max_num_seqs` 设为 8；它们可按实测调整。`kv_cache_memory_bytes` 可以显式覆盖缓存预算，但默认是 null，不是永久固定 8GiB。32 条采样是每步总量，不要求32条全部同时驻留。

物理显存验收按实际整卡总量减去余量计算，默认预留1GiB；不能继续硬套旧的38/76阈值，也不要把 PyTorch allocated 当成整张卡的物理峰值。短链路只覆盖它实际生成的序列，正式最大长度仍要另验。这些是 §6.32–6.33 的历史修正，本章不会把后来的门禁写成起初预先声明的标准。

### 8.3 配额的名字比“有一张 GPU”更具体

round9 的 CPU 沙箱阶段先后遇到四个提交前后错误：

| 次序 | 实际问题 | 正确解释 |
| --- | --- | --- |
| 1 | 取错项目编号字段 | Compute 项目的 id 不能替代 Cloud 项目编号；本地校验阻止继续 |
| 2 | 给标准 E2 配置不适用的维护策略 | 机器配置无效，不是 GPU 缺货；已建防火墙随后删除 |
| 3 | 请求16 vCPU，但全局 CPU 限额为12 | 区域还有额度不等于全局额度也够 |
| 4 | 500GB pd-balanced 超过 SSD 限额250GB | pd-balanced 计入 SSD 配额，不能按普通盘配额判断 |

后续使用两个4 vCPU规格以及各100GB磁盘完成探针。每次失败都有清理核验，不把“没创建 CustomJob”误解成“肯定没有其他资源”。

另一次误判更值得记住：普通 Compute A100 配额申请被拒绝，曾被当作 CustomJob 也不可用。§6.46 修正为按 Vertex 的作业准入指标判断：SPOT 看抢占式 A100 指标，STANDARD 看非抢占式指标。当时前者限额8、使用0，后者限额0。普通 Compute A100 为0，不应直接否决该 Spot CustomJob；但 SSD 和相关 CPU 需求仍要分别核对。

![图 8-2 只读预检按资源与服务命名空间核算，不把 fit 当成资源预订](images/ch08_preflight.svg)

## 命令

### 8.4 只在 CPU 上重现“能否申请”的判定

本章不要求读者登录云账号、申请配额或创建机器。`examples/ch08_preflight.py` 是新写的离线例子，只接收字典，返回缺口；没有云 SDK、网络调用或提交入口。它的字段名是教学别名，不是可直接调用的 API 名。

```bash
python examples/ch08_preflight.py
pytest -q tests/test_ch08_preflight.py
```

下面片段使用历史配额数字构造本地快照。普通 Compute GPU 额度为0不干扰正确的 Vertex Spot 检查；300GB磁盘则确实失败。

```python
from examples.ch08_preflight import Request, deficits, eligible_to_request
limits = {'compute/global_cpu': (12, 0), 'compute/ssd_gb': (250, 0),
          'vertex/spot_a100': (8, 0), 'compute/a100': (0, 0)}
assert eligible_to_request(Request(), limits, dict(limits))
assert deficits(Request(worker_disk_gb=300), limits) == ['compute/ssd_gb: insufficient']
```

教学实现要求每个必要指标都有有限、非负的 limit 与 usage；未知不是0，也不是通过。这比历史 `fit_deficits` 对部分缺失 Compute 字段的处理更严格，故明确不是源函数的逐行复制。两份相同快照只能支持“目前看可申请”，不能锁住配额、保证库存或自动授权创建。正式提交仍须使用实际配置、实际环境与有效授权。

源预检对 Vertex GPU 的核心分支如下，`result/round10/scripts/preflight_fit.py` L48–54，提交 `ca57456`，连续原样摘录。它选择相应调度类型的指标，缺失指标直接列为缺口；本书不执行此云预检脚本。

```{.python .source-excerpt}
    count = int(spec.get("accelerator_count") or 0)
    if count:
        metric = VERTEX_A100[spec["strategy"]]
        quota = vertex_quotas.get(metric)
        if quota is None or quota["usage"] + count > quota["limit"]:
            deficits.append(metric)
    return deficits
```

### 8.5 镜像构建也需要失败分类

round8 的第一版镜像在 PEP 668 系统 Python 保护处失败；第二版改用固定的 uv 二进制后，停在可选 causal-conv1d 源码编译。最后可见日志之后长时间没有新输出，操作者主动取消了构建。证据不支持把它改写为 OOM、编译器错误或已证明死锁。

第三版显式跳过该可选扩展，沿用已有 PyTorch 卷积回退，保留其他所需组件并完成导入核验。导入 torch、Transformers、PEFT、vLLM 与 FLA 成功，并非对应 GPU 内核已经跑过。固定镜像摘要、依赖版本与实际能力测试仍是后续合同。

![图 8-3 round8 三次构建观测墙钟时长，含失败和主动取消，不是账单时长](images/plots/ch08_builds.png)

图中原始秒数为255.836、1590.938、1579.385，合计3426.159秒，约57.1分钟。图表保留失败和取消，不只展示最后一次成功来缩短工程历程。绘图数据为 `data/ch08_builds.json`，只摘录状态和时长，没有云资源标识。

### 8.6 环境搬家，不重写旧证据

round8 的资产导入不是把文件复制过去就算完成。数据 manifest、adapter、模型 revision 与原始配置都要逐项匹配。一次配置哈希不符，最终从固定 Git 对象恢复了历史原字节，而不是把新文件的哈希写回旧 handoff 来消除警告。跨代码版本的关联记录另存，原始训练交接保持不变。

![图 8-4 冻结资产跨环境导入的证据链，不能替代运行和质量门禁](images/ch08_import.svg)

同一份 adapter 可以“文件完整”但“目标运行环境尚未验证”，也可以“能加载”但“质量报告未通过”。平台入口应保留这种状态组合，不把导入成功自动转换成允许正式 RL。这里讲的是流程合同，不对后来的质量阈值调整作预先承诺。

## 时间线与耗时

下表将 UTC 和 SGT 并列。没有时分的日级记录不补造时刻；作业排队、执行、构建与代码改造也不互相混算。

| 记录 | UTC | SGT（UTC+8） | 能证明什么 |
| --- | --- | --- | --- |
| round3 直接 VM 方案 | 仅方案，无启动时刻 | 未运行 | 没有可填的实际耗时 |
| round4 提交 | 10-13 11:52:25 | 10-13 19:52:25 | 请求进入平台 |
| round4 首条探针日志 | 10-13 12:06:45 | 10-13 20:06:45 | 前面约14分钟不是训练耗时 |
| round4 结束 | 10-13 12:07:21 | 10-13 20:07:21 | 脚本结束，不代表 Docker 子项成功 |
| round5 第二次 shell 探针 | 10-13 12:38:27–12:43:27 | 10-13 20:38:27–20:43:27 | 约5分钟测试窗口 |
| round6–7 本地平台适配 | 10-13，日级记录 | 原记录按日期归档 | 无云资源启动，不估算云时长 |
| round8 三次构建 | 10-14 12:10:11 起至13:12:38 | 10-14 20:10:11 起至21:12:38 | 三次运行时长之和不含中间间隔 |
| round9 成功探针创建 | 10-14 13:48:17 | 10-14 21:48:17 | 后续约18分钟才到 RUNNING |
| round9 RUNNING→SUCCEEDED | 10-14 14:06:10–14:10:14 | 10-14 22:06:10–22:10:14 | 4分4秒状态区间，不是模型 RL |
| 两次正确口径只读预检 | 10-14 18:11:23、18:11:47 | 10-15 02:11:23、02:11:47 | 跨日，两次快照间隔24秒 |

round6–7 的工作产物是平台适配、配置校验与 CPU 测试，而不是 GPU 成绩。round8 完成数据与两种尺寸 adapter 的导入、配置恢复和哈希核验；round9 才把原始任务容器的执行链路接通。不要用轮次编号暗示每一轮都有一次完整模型训练。

## 坑与修复

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| 作业 SUCCEEDED，Docker 却不可用 | 主脚本完成与子能力结果不同；层注册被禁 | 保留失败子项，改由临时 CPU VM 承担原任务容器 | round4：`failed to register layer: unshare: operation not permitted`（`docs/apexo/00_progress.md` 台账）；决策 R44 |
| 40GB 短链路能跑，旧门禁却失败 | 逻辑 allocated 与物理占用混用 | 分离统计口径、按实际物理容量与余量验收 | `docs/apexo/00_progress.md` §6.32–6.33 @ ca57456 |
| 配额看似充足仍提交失败 | 全局/区域、CPU/SSD、Compute/Vertex 混淆 | 读实际配置，逐资源与命名空间核对两次 | round9 两次只读预检快照（§6.42、§6.44–6.46） |
| 第二版镜像长时间无输出 | 可选 causal-conv1d 进入源码编译；故障未证实 | 主动取消；第三版显式回退 | R47；`data/ch08_builds.json` 中 CANCELLED 构建墙钟 1590.938s |
| 旧交接配置哈希不符 | 手工改写文件并非历史原件 | 从固定 Git 对象恢复原字节 | round8 配置恢复记录 @ ca57456 |
| 删除请求结束仍可能计费 | 资源清单与软删除/保留存储口径不同 | 读回临时资源不存在，另列持久存储 | `result/round8/costs-and-retention.json`（`currency_amount=null`） |

## 花费

作者 10-19 回顾给出全项目 GCP 约200美元、不含 Colab。这是项目总额口径，不是本章几轮的分项账单。`result/round8/costs-and-retention.json` 的 `currency_amount` 为 null，并明确构建观测时长不是计费时长；因此本书不把57.1分钟乘一个猜测单价。

该文件同时记录：round8 没有创建 GPU job 或 CPU VM，但仍保留对象与镜像存储，且继续产生存储费用。软删除7天也并非立即物理回收。“没有正在运行的机器”推不出“没有费用”。

## 边界说明

【边界说明】本章只建立计算与执行底座；预检、构建墙钟和清理读回都不能当成模型能力或计费账单。

## 教训

把平台工作拆成三个问题：是否具备能力、是否有正确的准入额度、失败后是否真正回收。每个问题都需要自己的证据。只读预检宜早做，但它替代不了实际运行；运行成功宜细分，但它替代不了任务质量；清理宜读回，但它不能抹去保留资产的成本。

本章到此只建立真实 RL 所需的计算与执行底座。下一章解释一个更新步内部的数学和权重同步，后面的训练结果章节才讨论真实奖励与门禁，不提前把基础设施探针包装成模型能力提升。

## 本章出处

- `docs/apexo/07_gcp_colab_a100_training_plan.md`、`docs/apexo/10_agent_platform_plan.md`、`docs/apexo/11_timeline.md`，提交 `ca57456`。
- `result/round3/README.md` 至 `result/round9/README.md`，提交 `ca57456`；其中 round3 明确未运行。
- `docs/apexo/00_progress.md` §6.22、§6.32–6.46；`docs/apexo/06_risks_and_decisions.md` R40、R44、R47，提交 `ca57456`。
- `result/round8/costs-and-retention.json`、`result/round10/scripts/preflight_fit.py`，提交 `ca57456`。本章只发布脱敏数值和原创离线练习，不发布云配置、身份或资源链接。
