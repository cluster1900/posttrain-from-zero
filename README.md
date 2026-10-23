# 后训练从 0 到 1

作者：**Hawk Wu**

[在线阅读](https://cluster1900.github.io/posttrain-from-zero/) · [下载 EPUB](https://cluster1900.github.io/posttrain-from-zero/posttrain-from-zero.epub) · [下载 PDF](https://cluster1900.github.io/posttrain-from-zero/posttrain-from-zero.pdf) · [Releases（EPUB / PDF / 离线网页包）](https://github.com/cluster1900/posttrain-from-zero/releases)

一本基于真实项目的中文后训练书：在官方 `Qwen/Qwen3.5-2B-Base` 上，用作者自行设计的 **Apexo 后训练流程**完成数据冻结 → LoRA SFT → outcome-only RL（DPPO）→ Terminal-Bench 2.0 全量评测的完整过程。第 1 章讲 PyTorch 基础，之后按时间顺序讲每一步的做法、原因、踩过的坑、解决办法、耗时和调整。

> **状态**：写作中。已交付前言与第 1–7 章；本批继续第 8–10 章的算力平台、单卡 DPPO 与沙箱执行合同；其余章节见[详细大纲](docs/outline.md)，尚未交付。

## 构建

全书一次构建出三种格式：网页 `dist/index.html`、EPUB `dist/posttrain-from-zero.epub`、PDF `dist/posttrain-from-zero.pdf`（A4，目录、页码、PDF 书签，中文字体嵌入）。正文里的 ```` ```mermaid ```` 图在构建时预渲染成 PNG，三种格式显示同一张图。

```bash
# 依赖：pandoc ≥ 3.0、Node ≥ 22.13、Chrome/Chromium、Noto CJK 字体、poppler-utils
PUPPETEER_SKIP_DOWNLOAD=1 npm ci                    # mermaid-cli + puppeteer（用系统 Chrome）
python3 scripts/check_book.py                      # 审核门禁：命名、保密、章节结构、出处、配图、示例测试
# 先按下方“CPU 教学例子”安装依赖
python3 -m pytest -q tests                        # CPU 示例测试
python3 build.py                                   # 生成网页、EPUB、PDF（找不到 Chrome 时用 CHROME_PATH 指定）
python3 scripts/check_built_book.py                 # 正文图片与独立教学代码
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

## CPU 教学例子

默认教学例子不下载模型、不创建云资源；第 6 章另列的可选配置下载入口不属于离线默认路径；教学环境与源项目 GPU 环境分开。

```bash
python3 -m venv .venv-examples
. .venv-examples/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-examples.txt
python examples/ch01_pytorch.py all
python -m pytest -q tests
for f in scripts/draw_ch*.py; do python "$f"; done
for f in scripts/figures/*.py; do python "$f"; done
python scripts/check_book.py
python3 build.py
python scripts/check_built_book.py
```

核验范围见 [批次 1 验证记录](docs/batch1-validation.md)；事实来源见 [批次 1 证据索引](docs/batch1-evidence.md)。

第二批事实映射与验证边界见 [批次 2 证据索引](docs/batch2-evidence.md) 和 [批次 2 验证记录](docs/batch2-validation.md)。

第三批事实映射与验证边界见 [批次 3 证据索引](docs/batch3-evidence.md) 和 [批次 3 验证记录](docs/batch3-validation.md)。

第四批事实映射与验证边界见 [批次 4 证据索引](docs/batch4-evidence.md) 和 [批次 4 验证记录](docs/batch4-validation.md)。本批三个 CPU 例子不连接云平台、不执行模型命令、不创建容器。
