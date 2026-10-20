# 第 10 章 沙箱：让模型的命令只在该在的地方跑

## 动机

终端代理的输出会变成程序执行。模型写出 `rm`、重定向、后台任务或环境变量赋值时，系统不能靠“它应该只是在解题”维持安全边界。项目合同要求：模型命令进入任务容器，不在训练 worker 或控制机上裸跑；模型也不应获得控制云资源的凭据。

这不是“用了 Docker 所以绝对安全”的结论。容器共享宿主内核，网络、挂载、权限和资源限额仍会影响风险；CPU VM 上的可信控制程序也拥有管理容器的能力。实际隔离范围必须逐层说明，不能把一个原生 shell 探针升级成完整安全审计。

本章从 round4 的 Docker 能力失败、round5 的原生 shell 局部验证，走到 round9 的远程原任务容器探针。依据为 `train/sandbox.py`、`train/remote_sandbox.py`、`train/agent_env.py`、平台合同与对应 README，提交 `ca57456`。

![图 10-1 信任边界：可信控制面与模型命令分开；箭头不表示凭据进入任务容器](images/ch10_boundary.svg)

## 配置

### 10.1 为什么不把 Docker 直接塞进 GPU worker

round4 在 CustomJob 中装上了 Docker CLI，也启动了私有 daemon，`docker info` 成功。但 BusyBox 镜像层注册报 `unshare: operation not permitted`，拉取失败，容器命令没有运行。这证明“客户端和 daemon 可启动”不等于平台允许所需的嵌套容器能力。

项目没有把这次限制当成提高权限的理由，而是把 GPU worker 留给训练与推理，原始任务容器移到每次运行独立的临时 CPU VM。GPU worker 通过 IAP SSH 发送可信容器控制命令，模型字符串作为容器内 bash 的参数传入。R44 记录宿主不挂服务账号，IAP 密钥不进入容器或成果；R40 要求沙箱只接收单个任务文件，不拿共享存储凭据。

这些是源项目设计与运行合同，不是本书已经做过渗透测试的证明。当前书中不会创建 SSH 身份、开放端口、配置 sudo、启动 Docker 或修改云权限。

### 10.2 把命令看成数据，直到进入指定容器

`RemoteHost.run` 处理的是可信宿主操作参数；它用 `shlex.join` 组织远程命令。模型文本经过容器执行包装，最终在容器内部被 bash 解释。宿主层不应把模型文本拼成另一条裸 shell 命令，也不允许模型选择容器名、SSH目标或 Docker 的权限参数。

`train/sandbox.py` L147–153，提交 `ca57456`，连续原样摘录如下。这里只展示边界代码，不执行：

```{.python .source-excerpt}
    def exec(self, command: str, timeout: int) -> ExecResult:
        if not self.container:
            raise SandboxError("container not started")
        wrapped = f"timeout --signal=TERM --kill-after=10 {int(timeout)} bash -c {shlex.quote(command)}"
        started = time.monotonic()
        try:
            res = self._docker(["exec", self.container, "bash", "-c", wrapped], timeout=timeout + 60)
```

单引号或分号仍然可以是合法模型命令的一部分，关键不是删除这些字符，而是它们在哪一层被解释。正确引用保护外层参数边界，不负责把任意命令变成无害命令。Python 的 [shlex 文档](https://docs.python.org/3.12/library/shlex.html) 明确限定 Unix shell 语义；不能把同一引用方案直接推广到所有操作系统和 shell。

归档 RL 配置给任务容器4g内存、2 CPU、1024个 PID 和 bridge 网络。它不是默认断网沙箱，也没有依据声称完成了所有出站网络限制。`run_args` 设置内存/交换上限、CPU、PID与入口；这些限制缓解资源滥用，不构成防逃逸证明。任务镜像与共享内核的可信度还需在部署环境中单独评估。

### 10.3 文件还在，不代表 shell 会话还在

round5 原生 shell 探针验证了跨调用文件持久化，但新一次调用的 cwd 和环境变量会复位。对于需要多步编译或切换目录的代理，若直接把每个调用当成独立新 shell，任务语义就变了。

项目环境包装器把 cwd 和导出的变量保存到容器内状态文件，下一轮载入，再执行本轮命令。它只在控制流能走到保存语句时记录状态；若命令直接退出 shell 或被杀，不能承诺完整保存。状态文件位于任务容器，不是把用户主机环境交给模型。

![图 10-2 多轮会话状态显式保存；超时或提前退出可能跳过保存步骤](images/ch10_state.svg)

`TerminalTaskEnv.reset` 先启动环境，准备目录与该任务的 seeds，再写入包装器。测试目录在提交验证时单独上传，奖励文件先清除旧值，防止把上一次结果误当这次成绩。这里没有把全量任务库或云存储凭据挂到容器里。

一次模型响应也可能包含多个工具调用。历史改动 `a5a8046` 将它们按顺序执行，每个调用消耗一次行动预算，遇到提交即停止；观测按调用顺序返回。不能把模型一条消息与一个环境动作等同。

### 10.4 exit code、超时和 reward 是三种信号

普通命令的退出码反馈行动是否执行成功。验证脚本退出码与任务奖励又不是同一信号：源合同以 reward 文件评分，非零测试退出码不应覆盖验证器已经合法写出的奖励。反过来，exit=0也不代表reward=1。

验证器超时或被 OOM 杀死时，源实现将该次 score 置0，不相信残留奖励文件；无法读取或解析奖励也得到0，并保留相应诊断字段。外围执行异常另由轨迹合同标成基础设施错误，进入第 9 章的过滤路径。不能把所有退出异常统一改写成 infra，也不能静默把通信失败当普通任务失败。

round9 正是一个清楚的反例：两个空提交都记录 `test_exit=0`、`reward=0.0`，探针阶段通过。它通过的是拉镜像、reset、执行、提交、测试和读取奖励这条链路，不是模型解决了两个任务，也没有证明正确解法一定得到1分。

![图 10-3 探针链路通过与任务奖励通过分开记录；空提交0分不是成功解题](images/ch10_verifier.svg)

## 命令

### 10.5 不执行命令也能检查外层合同

本章 `examples/ch10_sandbox_contract.py` 是参数和状态模拟器。它只返回列表与分类字符串，不导入 subprocess，不创建容器，也不执行模型文本。

```bash
python examples/ch10_sandbox_contract.py
pytest -q tests/test_ch10_sandbox_contract.py
```

下面调用展示两点：带引号的文本仍作为一个内层参数保留；验证器退出0而奖励0，只能判为任务失败。

```python
import shlex
from examples.ch10_sandbox_contract import container_argv, Probe, result_kind
command = 'echo "a b"; exit 7'
argv = container_argv('teaching-container', command)
assert shlex.split(argv[-1])[-1] == command
assert result_kind(Probe(0, 0.0)) == 'task failed'
assert result_kind(Probe(7, 1.0)) == 'task passed'
```

测试还放入分号、换行、命令替换等字符串，只检查它们保持在容器命令参数里，不会执行它们。容器名与超时由可信调用方提供并校验。这个例子证明的是参数构造合同，不是容器逃逸、恶意镜像或网络隔离的安全证明。

教学分类器把超时、OOM或没有合法二元奖励称为“verifier incomplete”，便于区分证据状态；它不是源函数的评分替代品。源函数在这些情形按其合同返回0，本书没有修改那份历史策略。

### 10.6 删除完成要有读回结果

round5 删除父资源后，父资源和 sandbox 的 GET 都返回404，但模板仍是 ACTIVE。第一次清理检查因此失败，显式删除模板后最终才通过。这说明“级联删除”也需要枚举合同中所有资源。

round9 记录的最终清理不仅看控制器退出码，还读回实例、磁盘、防火墙和作业清单；临时对象与本地私钥也有处理记录。书中只保留结果，不公开实际名称、路径中的资源标识或连接参数。

![图 10-4 清理是请求、等待、逐项读回与分类；未知状态不能写成 VERIFIED](images/ch10_cleanup.svg)

教学 `cleanup_verified` 接收四种预期资源的读回状态，缺一项、任意UNKNOWN或仍存在都返回False。它不是云平台客户端，也没有自动重试删除。源本地 DockerBackend 的 close 与远程 backend 的错误处理也不完全相同，因此不能声称调用任意 `close()` 就等于整个作业已回收。

## 时间线与耗时

| 记录 | UTC | SGT（UTC+8） | 结果边界 |
| --- | --- | --- | --- |
| 顺序多工具调用提交 `a5a8046` | 10-12 01:42:44 | 10-12 09:42:44 | 代码提交时间，不是任务执行耗时 |
| round4 GPU worker 容器探针 | 10-13，探针日志12:06:45起 | 10-13 20:06:45起 | daemon可用，容器层注册失败 |
| round5 第二次原生 shell 探针 | 10-13 12:38:27–12:43:27 | 10-13 20:38:27–20:43:27 | 基本命令和文件状态通过，创建约18.9秒 |
| round5 最终清理完成 | 10-13 12:45:06 | 10-13 20:45:06 | 模板、sandbox与父资源404 |
| round7 远程容器适配 | 日级记录10-13 | 按源日期归档 | 本地代码与假后端测试，无实际云作业 |
| round9 成功探针运行区间 | 10-14 14:06:10–14:10:14 | 10-14 22:06:10–22:10:14 | 两个原任务空提交0分，链路通过 |

round5 的 `sleep 5` 配1秒超时，API以HTTP400、FAILED_PRECONDITION及DEADLINE_EXCEEDED报告命令被终止；通用客户端把它向外抛出，探针进程退出1。不能把API报告“已杀死”翻译成当时端到端超时映射全部通过。该轮原始报告仍为局部能力验证，原任务验证、跨沙箱隔离以及模型 RL 都是NOT_RUN。

## 坑与修复

| 现象 | 根因 | 修复 | 证据 |
| --- | --- | --- | --- |
| Docker info 成功但容器不能跑 | 平台禁止所需的 unshare 操作 | GPU worker不承接原任务容器，转临时CPU VM | round4、R44 |
| shell 第二次调用丢目录/变量 | 文件持久化不等于进程会话持久化 | 显式状态包装器，承认中断保存边界 | round5、`_wrapper_script` |
| 探针超时导致通用客户端报错 | 服务的超时状态尚未映射为统一结果 | 保留失败报告，要求后端映射与清理验证 | round5 summary |
| 把任意124退出都当超时 | 命令本身也可合法返回124 | 结合外层计时与终止信号，不只看退出码 | `DockerBackend.exec` |
| 把容器停止当OOM | Running=false不说明为什么停止 | 检查 OOMKilled 字段；不掩盖普通退出 | `DockerBackend._oom_killed` |
| 奖励文件与退出码冲突 | 测试完成状态和任务标签混用 | 按奖励合同评分，超时/OOM不信任残留标签 | `TerminalTaskEnv._run_tests` |
| 父资源删除后仍有模板 | 清理集合不完整 | 扩大读回核验到实际创建的全部临时资源 | round5最终清理 |

## 花费

round5 没有GPU运行，round9是CPU环境探针；但这不足以推出它们免费。存储、作业与网络的分项账单没有在本章依据中完整提供。本书只报告运行区间和资源回收状态，不编造价格。已软删除或明确保留的对象仍须按真实账单核对。

## 教训

先定义命令能到哪里、能带什么状态、能拿什么凭据，再讨论模型能执行哪些工具。进程退出、基础设施健康、验证器完成、任务奖励和资源清理应该是分别记录的字段，而不是一个绿色勾。

沙箱探针的价值在于缩小未知范围。round4排除了GPU worker内嵌容器路线，round5说明原生shell的部分能力与缺口，round9才证实原任务验证链路能走通。它们都不能替代后面的真实模型训练与质量评测。

## 本章出处

- `train/sandbox.py`、`train/remote_sandbox.py`、`train/agent_env.py`、`scripts/agent_platform/cpu_startup.sh`，提交 `ca57456`。静态摘录按正文行号标记，不执行启动或权限脚本。
- `result/round4/README.md`、`result/round5/README.md`、`result/round5/summary.json`、`result/round7/README.md`、`result/round9/README.md`，提交 `ca57456`。
- `docs/apexo/08_execution_contract.md`、`docs/apexo/10_agent_platform_plan.md`、`docs/apexo/06_risks_and_decisions.md` R40与R44，提交 `ca57456`。
- Python `shlex` 官方文档见正文。原任务数据、harness与基准的外部归属见许可附录，本章不重新发布镜像、任务或验证脚本。
