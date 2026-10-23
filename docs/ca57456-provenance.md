# 第 9 章 ca57456 来源记录

本页记录第 9 章真实曲线使用的固定实验快照。它只保存可公开复核的提交号、相对路径和字段名，不保存账号、主机、云项目或存储标识。

## 固定来源

- 来源标签：固定实验提交 `ca57456`
- 原始相对路径：`result/round2/results/rl_2b_24k_metrics.jsonl`
- 书稿筛选文件：`data/ch09_rl_2b_24k_metrics.json`
- 选取记录：step 1–3，2B/24K 指标
- 绘图入口：`scripts/figures/ch09_rl_2b_24k.py`
- 输出图：`assets/images/plots/ch09_reward_useful.svg`、`ch09_kept_frac.svg`、`ch09_sync_diff.svg`、`ch09_time_split.svg`

## 字段映射

| 书稿字段 | 快照字段 | 用途 |
| --- | --- | --- |
| reward_mean | `rollout/reward_mean` | reward 曲线 |
| useful_groups / groups | `groups/useful_groups`, `groups/groups` | 有效组比例 |
| kept_frac | `train/kept_frac` | DPPO 保留比例 |
| sync_logprob_abs_diff | `sync/logprob_abs_diff` | 同步漂移 |
| gen_s / train_s / step_s | `time/gen_s`, `time/train_s`, `time/step_s` | 时间拆分 |

## 复核命令

在包含本页相对路径的仓库副本中运行：

```bash
python3 scripts/figures/ch09_rl_2b_24k.py
python3 -m unittest discover -s tests -v
```

脚本从 JSON 读取记录并重新计算 SVG 坐标；测试会核对提交号、记录范围、正文数字和单数字变异。没有重新启动训练，也没有把本页数据外推为 round10、rehearsal 或生产吞吐证据。
