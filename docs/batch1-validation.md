# 批次 1 验证记录

验证日期：2026-10-19。范围仅第 0–1 章及配套教学代码。本文记录本地证据；远程 CI 以对应 Draft PR 的最新 commit 检查为准，不预先宣称通过。

## 已通过

- 隔离 CPU 环境：Linux x86_64，Python 3.12.14，torch 2.10.0+cpu，transformers 5.6.0，peft 0.18.1，tokenizers 0.22.2；CUDA build 为 None，CUDA 不可用。
- `python examples/ch01_pytorch.py all`：六节断言全部通过。覆盖张量、累积、断点恢复、bf16/checkpoint/LoRA、移位 mask、纯本地 Transformers。
- 禁止 Python socket 建立外连的额外检查中，六节均通过。依赖安装需要网络，但示例运行不下载模型。
- `python -m pytest -q tests`：11 passed（6.48s），包括六节参数化测试、禁止外连的 CLI、正文六段代码逐字一致/执行检查与命名门禁回归。
- `python -m pip check`：无依赖冲突。
- `python scripts/draw_ch01.py`：生成两张确定性 SVG 示意图，非实测曲线。
- `python3 build.py`：Pandoc 3.1.11.1 成功生成 HTML 与 EPUB。
- EPUBCheck 5.3.0 本地验证：valid=True，messages=[]（通过 PyPI epubcheck 包所带检查器调用）；CI 另运行发行版 epubcheck。
- `python scripts/check_built_book.py`：HTML/EPUB 结构、两张嵌入图、六段正文 Python 代码全部通过。

## 已知提示与边界

- 本地 Pandoc 提示缺少 zh-CN 的 Abstract 翻译，但构建退出码为 0；标题、作者和中文正文不受该缺省标签提示影响。本书不含摘要节。
- 教学环境有意与源项目锁定的 GPU 栈区分。测试不是源项目 torch 2.10 / transformers 5.6 / PEFT / vLLM 的替代验收。
- 随机微型模型用于接口与张量断言，不具有预训练能力；不下载正式 Qwen 权重，不运行云任务或真实终端沙箱。
- 未做 GPU、分布式、吞吐、显存峰值、正式训练结果复现，也未进行所有 EPUB 阅读器的兼容性测试。
- 更新到大纲 v3 后，保留上游 `scripts/check_book.py` 作为构建前门禁；只修正其把标准算子 softmax 误判成旧名称的子串误报，并有正反回归测试。
- HTML 与 EPUB 的结构、资源嵌入、目录和源码示例另由 `scripts/check_built_book.py` 检查。远程 CI 会重跑教学例子与构建，但本分支不会部署 Pages。
