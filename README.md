# 后训练从 0 到 1

作者：**Hawk Wu**

[在线阅读](https://cluster1900.github.io/posttrain-from-zero/) · [下载 EPUB](https://cluster1900.github.io/posttrain-from-zero/posttrain-from-zero.epub) · [下载 PDF](https://cluster1900.github.io/posttrain-from-zero/posttrain-from-zero.pdf) · [Releases（EPUB / PDF / 离线网页包）](https://github.com/cluster1900/posttrain-from-zero/releases)

一本基于真实项目的中文后训练书：在官方 `Qwen/Qwen3.5-2B-Base` 上，用作者自行设计的 **Apexo 后训练流程**完成数据冻结 → LoRA SFT → outcome-only RL（DPPO）→ Terminal-Bench 2.0 全量评测的完整过程。第 1 章讲 PyTorch 基础，之后按时间顺序讲每一步的做法、原因、踩过的坑、解决办法、耗时和调整。

> **状态**：写作中。目前只有前言和[详细大纲](docs/outline.md)。

## 构建

全书一次构建出三种格式：网页 `dist/index.html`、EPUB `dist/posttrain-from-zero.epub`、PDF `dist/posttrain-from-zero.pdf`（A4，目录、页码、PDF 书签，中文字体嵌入）。正文里的 ```` ```mermaid ```` 图在构建时预渲染成 PNG，三种格式显示同一张图。

```bash
# 依赖：pandoc ≥ 3.0、Node ≥ 22.13、Chrome/Chromium、Noto CJK 字体、poppler-utils
PUPPETEER_SKIP_DOWNLOAD=1 npm ci                    # mermaid-cli + puppeteer（用系统 Chrome）
python3 scripts/check_book.py                      # 审核门禁：命名、保密、章节结构、出处、配图、示例测试
pip install -r requirements-examples.txt && pytest -q tests   # 有 examples/ 时：CPU 示例测试
python3 build.py                                   # 生成网页、EPUB、PDF（找不到 Chrome 时用 CHROME_PATH 指定）
python3 scripts/check_outputs.py dist              # 检查三种产物：下载链接、EPUB 图片、PDF 字体嵌入与页码
python3 build.py --src tests/build_fixture --out build/fixture   # 构建自检样章（中文、mermaid、公式、代码、表格）
epubcheck dist/posttrain-from-zero.epub            # 可选：本地校验 EPUB（CI 必跑）
```

每个 PR 都要满足[大纲里的验收标准](docs/outline.md)。

推送到 `main` 后，GitHub Actions 会自动构建并部署到 GitHub Pages（网页顶部有 EPUB / PDF 下载链接）；推送 `vX.Y.Z` 标签会把 EPUB、PDF、离线网页包和 `SHA256SUMS.txt` 附到对应 Release。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `chapters/` | 章节 Markdown，按文件名排序合并 |
| `docs/outline.md` | 详细大纲、写作规则、章节写法、验收标准、每章对应的源材料 |
| `examples/`、`tests/` | 能在 CPU 上运行的代码示例及其 pytest 测试 |
| `assets/` | 样式与图片（`assets/images/`） |
| `scripts/` | `check_book.py` 审核门禁、`check_outputs.py` 产物检查、`html2pdf.mjs` PDF 打印、`figures/` 画图脚本 |
| `tests/build_fixture/` | 构建自检样章（不进正式书稿） |
| `package.json` | 构建用的 Node 依赖（mermaid-cli、puppeteer），版本锁在 `package-lock.json` |
| `build.py` | 构建脚本：mermaid 预渲染 → Pandoc 出网页和 EPUB → Chrome 打印 PDF |
| `CHANGELOG.md` | 中文更新日志 |
