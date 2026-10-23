# 第四批验证记录

## 范围

第8–10章、三个离线CPU例子、11张原创中文SVG示意图、1张历史构建时长PNG、三个带行号静态源码摘录。原有章节、构建与权限边界保留。数据图尺寸逐图精确断言，不以降低全书字号解决长字段排版。

## 可复现入口

```bash
python -m pytest -q tests
python scripts/check_book.py
for f in scripts/draw_ch*.py; do python "$f"; done
for f in scripts/figures/*.py; do python "$f" --out build/figcheck; done
python3 build.py
python scripts/check_built_book.py
python scripts/check_outputs.py dist
epubcheck dist/posttrain-from-zero.epub
```

## 本地已完成

- Python 3.12、torch 2.10.0+cpu 的隔离环境，全套119项通过；三个新增CLI实际运行，全部13段正文教学Python fence通过。静态源码摘录单独逐字核验，不执行。
- 新图已实际栅格化/查看中文像素；构建时长图左标签排版修正后完整显示，保持1920×1088与原始数值。数据图全部恢复精确尺寸断言。
- book gate与扩展隐私扫描通过；未改变源数据、原有工作流、权限或打印字号。
- 实际执行 `python3 build.py`，前言Mermaid渲染因 Chromium `socket() failed: Operation not permitted` 中止；没有降低沙箱或门禁，不将其记为本地三格式构建通过。

## 当前边界

CPU结果、独立审校与精确head远程产物记录在本批Draft PR中逐项更新。完整三格式构建和PDF页面检查只有取得实际产物后才记为通过；示意图可渲染不等于整本PDF已经验收。不声称完整浏览器或所有EPUB阅读器兼容性测试。

## 独立本地审校

- 离线全套119/119通过（16.94s）、book gate通过；三个静态摘录逐字匹配，两个历史提交的UTC/SGT核验正确。
- DPPO源合同及额外200组含数值clamp的标量loss/梯度对照、4680组Fraction中心化组合通过；构建三时长与null金额保持原值。
- 11张SVG与1张PNG实际像素检查通过，所有新增阶段章各4图；107个公共文本/代码/JSON/SVG的初轮扩展隐私扫描无命中，新增图与文字差量另核。
- 审校指出同步/清理异常保证应收窄，9.5已明确同步在try外、sleep自身失败可能阻止迁移，不夸大全路径恢复。来源、许可和未运行边界通过。
- 此为本地内容/代码/图像通过，不替代精确远程head的三格式构建与实际PDF页面检查。
