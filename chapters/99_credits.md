# 致谢与第三方许可（当前批次）

本书署名 Hawk Wu。Apexo 是项目流程名，不改变已有算法、公开数据、模型与工具的作者和许可。本书第 0–1 章当前涉及的技术与项目背景先在这里保留署名，不等附录 E 完成才补。

## 原项目的外部输入

以下固定版本与许可名称来自源仓库 `THIRD_PARTY.md`，提交 `ca57456`；这是对原项目使用情况的披露，不表示本书重新发布了数据、模型权重或 harness 源码。

| 外部项目 / 作者 | 原项目用途与固定版本 | 许可记录 |
| --- | --- | --- |
| Ai2，Hugging Face 数据集 `allenai/tmax-15k-open-instruct` | RL 任务；revision `7b090eca98bf351356bc1c64290c5c4a09f2f98c` | ODC-BY 1.0 |
| Ai2，Hugging Face 数据集 `allenai/tmax-sft` | SFT 轨迹；revision `9d6ab7471ffa2884b2ef0cd2b7e5e22a027ff1b4` | ODC-BY 1.0 |
| `hamishivi/tmax` 评测 harness | 提供 `Vanillux2Agent`；commit `6d3d606c0f9de6ceec9481f9ccc793936818fef0` | Apache-2.0 |
| Harbor / Terminal-Bench 2.0 维护者 | 评测框架与基准 | 源清单要求参阅各自仓库；本批不猜测统一许可证 |
| Qwen 团队，`Qwen/Qwen3.5-2B-Base` / `Qwen/Qwen3.5-4B-Base` | 基座模型，revision 在源配置中固定 | 源清单要求参阅各模型卡；不将数据许可套用到模型 |

## 本章技术背景

- PyTorch 团队：张量、autograd、nn.Module、checkpoint、交叉熵 API。正文链接官方文档。
- Hugging Face 团队及贡献者：Transformers、Tokenizers、PEFT 的加载与适配器接口。正文链接官方聊天模板文档。
- Edward J. Hu 等：LoRA: Low-Rank Adaptation of Large Language Models，https://arxiv.org/abs/2106.09685 。本章保留 LoRA 的公开技术来源，不称其由本项目首次提出。

配套教学代码与 SVG 是新写的小规模示例，并非上述项目源码的改名副本。本批不分发第三方数据、模型或 harness；如果后续批次摘录或分发第三方材料，必须继续检查相关许可证、版权声明与 NOTICE 要求，而非仅依此表推定已履行所有义务。
