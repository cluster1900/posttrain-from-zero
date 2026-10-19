# 批次 1 验证记录

验证日期：2026-10-19。范围仅第 0–1 章及配套教学代码。本文记录本地证据；远程 CI 以对应 Draft PR 的最新 commit 检查为准，不预先宣称通过。

## 首轮返修版已通过（整合三格式构建之前）

- 隔离 CPU 环境：Linux x86_64，Python 3.12.14，torch 2.10.0+cpu，transformers 5.6.0，peft 0.18.1，tokenizers 0.22.2；CUDA build 为 None，CUDA 不可用。
- `python examples/ch01_pytorch.py all`：六节断言全部通过。覆盖张量、累积、断点恢复、bf16/checkpoint/LoRA、移位 mask、纯本地 Transformers。
- 禁止 Python socket 建立外连的额外检查中，六节均通过。依赖安装需要网络，但示例运行不下载模型。
- `python -m pytest -q tests`：17 passed（7.07s），包括六节参数化测试、禁止外连的 CLI、正文六段代码逐字一致/执行检查与命名门禁 A1 输出回归、源码摘录不执行隔离、图表数据与引用检查。
- `python -m pip check`：无依赖冲突。
- `python scripts/draw_ch01.py`：生成五张确定性 SVG：四张示意图、一张项目实测指标对比图。
- `python3 build.py`：Pandoc 3.1.11.1 成功生成 HTML 与 EPUB。
- EPUBCheck 5.3.0 本地验证：valid=True，messages=[]（通过 PyPI epubcheck 包所带检查器调用）；CI 另运行发行版 epubcheck。
- `python scripts/check_built_book.py`：HTML/EPUB 结构、五张嵌入图、六段正文 Python 代码全部通过。

## 已知提示与边界

- 本地 Pandoc 提示缺少 zh-CN 的 Abstract 翻译，但构建退出码为 0；标题、作者和中文正文不受该缺省标签提示影响。本书不含摘要节。
- 教学环境有意与源项目锁定的 GPU 栈区分。测试不是源项目 torch 2.10 / transformers 5.6 / PEFT / vLLM 的替代验收。
- 随机微型模型用于接口与张量断言，不具有预训练能力；不下载正式 Qwen 权重，不运行云任务或真实终端沙箱。
- 未做 GPU、分布式、吞吐、显存峰值、正式训练结果复现，也未进行所有 EPUB 阅读器的兼容性测试。
- 更新到大纲 v4 后，保留上游 `scripts/check_book.py` 作为构建前门禁；只修正其把标准算子 softmax 误判成旧名称的子串误报，并有正反回归测试。
- HTML 与 EPUB 的结构、资源嵌入、目录和源码示例另由 `scripts/check_built_book.py` 检查。远程 CI 会重跑教学例子与构建，但本分支不会部署 Pages。

## 首轮审核修订验证

- 五张 SVG 均用 Inkscape 和已安装 Noto Sans CJK SC 栅格化，逐张检查实际中文像素；没有缺字、文字遮挡或裁切。LoRA 为并行求和，label 图按零基位置对齐。
- 六段静态源码摘录逐一对照 `ca57456`，数据集标识的唯一替换已在正文说明，原始署名保留在许可附录。
- 图表仅使用 `scripts/data/batch1_metrics.json` 的脱敏百分数摘录，逐值对应结项总结 §2.1，不重新计算或修改原始指标。
- `check_built_book.py` 根据正文图片引用数核验资源，不再写死两张；只执行教学代码，跳过带 `.source-excerpt` 的项目片段。

## 上游三格式构建整合（2026-10-19）

- 整合上游 `c6f9243` 的 PDF、Mermaid、构建样章与 v4 门禁；保留原有权限和仅非 PR 部署条件。Q1–Q10 属于单独 PR，本批不新增答案。
- 前言保留上游全书路线图，顺延为图 0-3；新增的五张 SVG 仍为四张示意图与一张实测图。路线图由上游 Mermaid 渲染链生成 PNG。
- 整合后 `check_book.py` 通过，17 项测试通过（6.90s）；新增 matplotlib 3.11.2 后 `pip check` 通过；锁定 Node 依赖 `npm ci` 成功。
- 本地完整 `python3 build.py` 已实际尝试，但系统 Chromium 在 Mermaid 阶段因沙箱不允许 `socket()` 启动失败。因此上面 HTML/EPUB 成功证据仅适用于整合前版本，不能视为本次三格式构建通过。未修改权限或绕过限制；本次 HTML/EPUB/PDF、EPUBCheck 与 PDF 字体/页码检查以同一 PR 最新 head 的远程 CI 为准。
- 兼容 Pandoc 内联 SVG：网页和打印样式将 `figure svg` 与图片一起约束为容器宽度，并保持纵横比；待远程 PDF 实际图页复核。
