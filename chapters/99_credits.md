# 致谢与第三方许可（当前批次）

本书署名 Hawk Wu。Apexo 是项目流程名，不改变已有算法、公开数据、模型与工具的作者和许可。本书第 0–10 章当前涉及的技术与项目背景先在这里保留署名，不等附录 E 完成才补。

## 原项目的外部输入

以下固定版本与许可名称来自源仓库 `THIRD_PARTY.md`，提交 `ca57456`；这是对原项目使用情况的披露，不表示本书重新发布了数据、模型权重或 harness 源码。

| 外部项目 / 作者 | 原项目用途与固定版本 | 许可记录 |
| --- | --- | --- |
| Ai2，Hugging Face 数据集 `allenai/tmax-15k-open-instruct` | RL 任务；revision `7b090eca98bf351356bc1c64290c5c4a09f2f98c` | ODC-BY 1.0 |
| Ai2，Hugging Face 数据集 `allenai/tmax-sft` | SFT 轨迹；revision `9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4` | ODC-BY 1.0 |
| `hamishivi/tmax` 评测 harness | 提供 `Vanillux2Agent`；commit `6d3d606c0f9de6ceec9481f9ccc793936818fef0` | Apache-2.0 |
| Harbor / Terminal-Bench 2.0 维护者 | 评测框架与基准 | 源清单要求参阅各自仓库；本批不猜测统一许可证 |
| Qwen 团队，`Qwen/Qwen3.5-2B-Base` / `Qwen/Qwen3.5-4B-Base` | 基座模型，revision 在源配置中固定 | 源清单要求参阅各模型卡；不将数据许可套用到模型 |

## 第 0–10 章技术背景

- PyTorch 团队：张量、autograd、nn.Module、checkpoint、交叉熵 API。正文链接官方文档。
- Hugging Face 团队及贡献者：Transformers、Tokenizers、PEFT 的加载与适配器接口。正文链接官方聊天模板文档。
- Edward J. Hu 等：LoRA: Low-Rank Adaptation of Large Language Models，https://arxiv.org/abs/2106.09685 。本章保留 LoRA 的公开技术来源，不称其由本项目首次提出。

配套教学代码与 SVG 是新写的小规模示例，并非上述项目源码的改名副本。本批不分发第三方数据、模型或 harness；如果后续批次摘录或分发第三方材料，必须继续检查相关许可证、版权声明与 NOTICE 要求，而非仅依此表推定已履行所有义务。

- Rafael Rafailov 等：Direct Preference Optimization，https://arxiv.org/abs/2305.18290 。
- John Schulman 等：Proximal Policy Optimization Algorithms，https://arxiv.org/abs/1707.06347 。
- Zhihong Shao 等：DeepSeekMath（GRPO），https://arxiv.org/abs/2402.03300 。
- Penghui Qi 等：Rethinking the Trust Region in LLM Reinforcement Learning（DPPO），https://arxiv.org/abs/2602.04879 。
- Python 贡献者：hashlib、tarfile 提取过滤器文档。

第 2 章仅独立解释上述公开算法的区别，未复制论文图表或长段文字；不声称本项目发明了这些算法。第 4 章发布的是项目统计的最小脱敏摘录，不是外部任务/轨迹内容。

## 项目源码摘录

第 1、3、4、5、6、7、8、9、10 章的静态源码摘录来自作者授权整理的项目，按正文逐项注明路径、行号和提交（结项 `ca57456`，历史版本 `047c932`／`fbf621e`），不是独立可运行示例。摘录保留代码中的接口和归属线索；其中一个数据集标识在正文作明示替换，原始公开数据来源仍见上表。源项目此版本未提供根目录 LICENSE，本书不据此推定新的开源软件许可，也不将 PyTorch、PEFT 或 LoRA 归为本项目原创。

第 5 章讨论的源聊天模板在其文件头注明基于 Qwen 官方模板（Apache-2.0），并记录了思考段策略调整；本书未复制该模板全文或外部系统提示词。模板/工具格式同时涉及上表所列 Ai2 数据与评测 harness，Apexo 名称不替代这些来源。第 6 章直接使用 Hugging Face Transformers 的 Qwen3.5 实现和 PEFT，在随机微型配置上统计参数，不将其模型架构或库代码声称为作者原创。

第 7 章的曲线来自项目训练日志数值摘录，不含外部轨迹内容。原始数值未平滑或重估；它们与第 6 章本书 CPU 读者实验的结果分别标记。

第 8 章构建时长图只含项目观测数值，不含云资源标识，不从时长推算账单。第 9 章具体解释 DPPO 的二元 TV 分支，算法归属仍为 Penghui Qi 等；项目组内采样与执行合同也保留上表中的参考实现背景。第 10 章的参数模拟器由本书新写，不复制任务镜像、原始验证脚本或外部系统提示词。Python、Docker、Google Cloud、vLLM 等工具名称仅指明实现依赖，不表示本书拥有这些工具的版权。
