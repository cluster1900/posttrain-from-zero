# 前言

> 本书正在写作中。当前只有大纲，完整目录见仓库 [docs/outline.md](https://github.com/cluster1900/posttrain-from-zero/blob/main/docs/outline.md)。

这本书记录一次真实的大模型后训练：在官方 Qwen3.5-2B / 4B 基座模型上，用监督微调（SFT）和强化学习（RL）把它训练成能在终端里调用 bash 完成任务的智能体，再用 Terminal-Bench 2.0 做评测。项目从 2026 年 10 月 10 日开始，10 月 19 日结项。整套方法叫 **Apexo 后训练流程**，是作者在阅读多方公开论文与开源实现之后自行设计和修改的。整个项目总共花了大约 200 美元，全部是 Google Cloud 的费用。

第 1 章先补上读懂后训练代码需要的 PyTorch 基础。从第 2 章起按时间顺序讲整个过程：每一步做了什么、为什么这样做、遇到了哪些坑、怎么解决的、花了多长时间、之后又做了哪些调整。书中的数字都来自项目的训练日志、指标文件和提交记录。

## 全书路线图

```{.mermaid #fig-roadmap caption="图 0-1　全书路线图：先打基础，再按时间顺序走完整个项目"}
flowchart TB
  subgraph A[基础]
    direction LR
    C1[第 1 章<br/>PyTorch 入门] --> C2[第 2 章<br/>后训练全景与路线]
  end
  subgraph B[准备]
    direction LR
    C3[第 3 章<br/>从 v1 到 v2] --> C4[第 4 章<br/>数据冻结] --> C5[第 5 章<br/>聊天模板与工具调用] --> C6[第 6 章<br/>模型结构与 LoRA 选型]
  end
  subgraph C[训练]
    direction LR
    C7[第 7 章<br/>SFT] --> C8[第 8 章<br/>算力] --> C9[第 9 章<br/>RL 原理与实现] --> C10[第 10 章<br/>沙箱]
  end
  subgraph D[评测与判定]
    direction LR
    C11[第 11 章<br/>评测] --> C12[第 12 章<br/>RL 实跑] --> C13[第 13 章<br/>成败判定]
  end
  subgraph E[复盘]
    direction LR
    C14[第 14 章<br/>结果与复盘] --> C15[第 15 章<br/>工程纪律]
  end
  A --> B --> C --> D --> E
```

作者：Hawk Wu
