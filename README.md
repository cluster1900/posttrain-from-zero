# 后训练从 0 到 1

作者：**Hawk Wu**

[在线阅读](https://cluster1900.github.io/posttrain-from-zero/) · [Releases（EPUB / 离线网页包）](https://github.com/cluster1900/posttrain-from-zero/releases)

一本基于真实项目的中文后训练书：在官方 `Qwen/Qwen3.5-2B-Base` 上，用作者自行设计的 **Apexo 后训练流程**完成数据冻结 → LoRA SFT → outcome-only RL（DPPO）→ Terminal-Bench 2.0 全量评测的完整过程。第 1 章讲 PyTorch 基础，之后按时间顺序讲每一步的做法、原因、踩过的坑、解决办法、耗时和调整。

> **状态**：写作中。本批补齐前言与第 1 章 PyTorch 入门；其余章节见[详细大纲](docs/outline.md)，尚未交付。

每个 PR 都要满足[大纲里的验收标准](docs/outline.md)。

## 构建

```bash
# 先按下方“CPU 教学例子”安装依赖；需要 pandoc ≥ 3.0
python3 scripts/check_book.py
python3 -m pytest -q tests
python3 build.py      # 生成 dist/index.html 和 dist/posttrain-from-zero.epub
epubcheck dist/posttrain-from-zero.epub  # 本地需安装；CI 必跑
```

推送到 `main` 后，GitHub Actions 会自动构建并部署到 GitHub Pages；推送 `vX.Y.Z` 标签会把 EPUB 和离线网页包附到对应 Release。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `chapters/` | 章节 Markdown，按文件名排序合并 |
| `docs/outline.md` | 详细大纲、写作规则、章节写法、验收标准、每章对应的源材料 |
| `assets/` | 样式与图片（`assets/images/`） |
| `scripts/` | 审核、构建验证与画图脚本 |
| `examples/`、`tests/` | CPU 教学例子及逐项 pytest 断言 |
| `build.py` | Pandoc 构建脚本 |
| `CHANGELOG.md` | 中文更新日志 |

## CPU 教学例子

例子不下载模型、不创建云资源；教学环境与源项目 GPU 环境分开。

```bash
python3 -m venv .venv-examples
. .venv-examples/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-examples.txt
python examples/ch01_pytorch.py all
python -m pytest -q tests
python scripts/draw_ch01.py
python scripts/check_book.py
python3 build.py
python scripts/check_built_book.py
```

核验范围见 [批次 1 验证记录](docs/batch1-validation.md)；事实来源见 [批次 1 证据索引](docs/batch1-evidence.md)。
