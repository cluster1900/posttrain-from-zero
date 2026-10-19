# 后训练从 0 到 1

作者：**Hawk Wu**

[在线阅读](https://cluster1900.github.io/posttrain-from-zero/) · [Releases（EPUB / 离线网页包）](https://github.com/cluster1900/posttrain-from-zero/releases)

一本基于真实项目的中文后训练书：在官方 `Qwen/Qwen3.5-2B-Base` 上，用作者自行设计的 **Apexo 后训练流程**完成数据冻结 → LoRA SFT → outcome-only RL（DPPO）→ Terminal-Bench 2.0 全量评测的完整过程。第 1 章讲 PyTorch 基础，之后按时间顺序讲每一步的做法、原因、踩过的坑、解决办法、耗时和调整。

> **状态**：写作中。目前只有前言和[详细大纲](docs/outline.md)。

## 构建

```bash
# 需要 pandoc ≥ 3.0
python3 build.py      # 生成 dist/index.html 和 dist/posttrain-from-zero.epub
```

推送到 `main` 后，GitHub Actions 会自动构建并部署到 GitHub Pages；推送 `vX.Y.Z` 标签会把 EPUB 和离线网页包附到对应 Release。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `chapters/` | 章节 Markdown，按文件名排序合并 |
| `docs/outline.md` | 详细大纲、写作规则、每章对应的源材料 |
| `assets/` | 样式与图片（`assets/images/`） |
| `scripts/` | 画图等辅助脚本 |
| `build.py` | Pandoc 构建脚本 |
| `CHANGELOG.md` | 中文更新日志 |
