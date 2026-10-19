#!/usr/bin/env python3
"""check_book.py：审核门禁里能自动检查的部分。CI 在构建前运行；本地用法：python3 scripts/check_book.py

检查项（对应 docs/outline.md「验收标准」）：
  A1 品牌：书稿中不出现旧方法名；外部数据集 / harness 的 ID 只允许出现在致谢与许可附录。
  A2 保密：书稿与大纲中不出现云项目编号、bucket、服务账户、镜像仓库地址、邮箱、Drive 文件夹 ID、私钥或 API key。
  A3 结构：训练阶段章节与专题章节（第 6、13 章）必须包含规定的小节和表头；每章（前言除外）必须有「本章出处」。
  A4 出处：「本章出处」里至少列出一条源仓库路径或提交号。
  A5 示例：examples/ 下每个 .py 示例都要有对应测试（tests/test_<示例名>.py）。
  A6 配图：第 1–15 章每章至少 2 张图（mermaid 代码块或 Markdown 图片）；图片引用的文件必须存在；图要有图注。
  A7 数据图：assets/images/plots/ 下每张图都要有同名画图脚本 scripts/figures/<名字>.py，脚本里声明 DATA_SOURCE。
  A8 作者问题：docs/author-answers.md 存在时，Q1–Q10 每题都有「记录依据」「常见做法与出处」「结论」三个小节，
     「常见做法与出处」里至少一个 http(s) 链接，「结论」第一行带【记录事实】【读者运行所得】【作者事后分析】标签之一。
命中时只打印文件名、行号和规则名，不打印命中的内容。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "t" + "max"
CREDIT_FILE = re.compile(r"(credits|appendix_e|99_)")
CREDIT_IDS = re.compile(r"allenai/" + OLD + r"-(15k-open-instruct|sft)|hamishivi/" + OLD, re.I)
SECRET_RULES = {
    "gcs-bucket": re.compile(r"gs://[a-z0-9]"),
    "service-account": re.compile(r"iam\.gserviceaccount\.com"),
    "artifact-registry": re.compile(r"-docker\.pkg\.dev/"),
    "gcp-project-id": re.compile(r"\bproject-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{3}\b"),
    "gcp-project-number": re.compile(r"(?<![\d.])\d{12}(?![\d.])"),
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "drive-folder-id": re.compile(r"\b1[A-Za-z0-9_-]{32}\b"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "api-key": re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{35}|ghp_[A-Za-z0-9]{36}|hf_[A-Za-z0-9]{30,})\b"),
}
STAGE_CHAPTERS = {"03", "04", "07", "08", "09", "10", "11", "12"}
STAGE_SECTIONS = ["动机", "配置", "命令", "时间线与耗时", "坑与修复", "花费", "教训", "本章出处"]
TOPIC_SECTIONS = {
    "06": ["动机", "模型结构", "检查方法与代码", "LoRA 目标模块与超参数", "冻结与训练的参数", "坑与修复", "设计理由与事后分析", "本章出处"],
    "13": ["动机", "信号总表", "评测信号", "训练信号", "门禁与判定", "实测值", "设计理由与事后分析", "本章出处"],
}
SIGNAL_HEADER = re.compile(r"\|\s*信号\s*\|\s*公式\s*\|\s*样本量\s*\|\s*数据来源\s*\|\s*原阈值\s*\|\s*修订阈值\s*\|\s*为什么这样算\s*\|\s*证据\s*\|")
PITFALL_HEADER = re.compile(r"\|\s*现象\s*\|\s*根因\s*\|\s*修复\s*\|\s*证据\s*\|")
MIN_FIGURES = 2
MERMAID_FENCE = re.compile(r"^(`{3,}|~{3,})\s*(mermaid|\{[^}]*\.mermaid\b[^}]*\})\s*$", re.M)
MERMAID_CAPTION = re.compile(r'caption="[^"]+"')
MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)[^)]*\)")
ANSWER_SECTIONS = ["记录依据", "常见做法与出处", "结论"]
ANSWER_LABEL = re.compile(r"【(记录事实|读者运行所得|作者事后分析)】")
SOURCE_REF = re.compile(r"`[^`]*(/|\.md|\.json|\.jsonl|\.py|\.yaml|\.log)[^`]*`|提交 `[0-9a-f]{7,40}`")


def md_files():
    yield from sorted((ROOT / "chapters").glob("*.md"))
    yield ROOT / "docs" / "outline.md"
    yield ROOT / "docs" / "author-answers.md"
    yield ROOT / "README.md"
    yield ROOT / "data" / "README.md"


def strip_code(text):
    """去掉非 mermaid 的代码块（逐行解析围栏），避免把代码里的 ![...] 当成图片。"""
    out, fence, keep = [], None, True
    for line in text.splitlines():
        m = re.match(r"^(`{3,}|~{3,})(.*)$", line)
        if fence is None and m:
            fence, keep = m.group(1), bool(MERMAID_FENCE.match(line))
            if keep:
                out.append(line)
            continue
        if fence is not None and line.startswith(fence) and not line[len(fence):].strip():
            if keep:
                out.append(line)
            fence, keep = None, True
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


def check_figures(rel, text, errors):
    body = strip_code(text)
    fences = MERMAID_FENCE.findall(body)
    images = MD_IMAGE.findall(body)
    for m in re.finditer(MERMAID_FENCE, body):
        if not MERMAID_CAPTION.search(m.group(0)):
            errors.append(f"{rel}: A6 mermaid 图缺少图注 caption=\"图 N-M　……\"")
    for alt, target in images:
        if not alt.strip():
            errors.append(f"{rel}: A6 图片缺少图注（![图注](路径)）")
        if re.match(r"https?://", target):
            errors.append(f"{rel}: A6 图片不能引用外链，放进 assets/images/")
        elif not any((base / target).exists() for base in (ROOT / "chapters", ROOT / "assets")):
            errors.append(f"{rel}: A6 图片文件不存在：{target}")
    n = len(fences) + len(images)
    if n < MIN_FIGURES:
        errors.append(f"{rel}: A6 只有 {n} 张图，每章至少 {MIN_FIGURES} 张")


def check_answers(errors):
    path = ROOT / "docs" / "author-answers.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    blocks = {m.group(1): m for m in re.finditer(r"^##\s+Q(\d+)\b.*$", text, re.M)}
    starts = sorted((m.start(), q) for q, m in blocks.items())
    for q in range(1, 11):
        if str(q) not in blocks:
            errors.append(f"docs/author-answers.md: A8 缺少 Q{q}")
    for i, (pos, q) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        block = text[pos:end]
        subs = {m.group(1).strip(): m.end() for m in re.finditer(r"^###\s+(.+)$", block, re.M)}
        for sec in ANSWER_SECTIONS:
            if sec not in subs:
                errors.append(f"docs/author-answers.md: A8 Q{q} 缺少小节「{sec}」")
        if "常见做法与出处" in subs:
            part = block[subs["常见做法与出处"]:].split("\n### ", 1)[0]
            if not re.search(r"https?://", part):
                errors.append(f"docs/author-answers.md: A8 Q{q}「常见做法与出处」没有链接")
        if "结论" in subs:
            first = next((l for l in block[subs["结论"]:].splitlines() if l.strip()), "")
            if not ANSWER_LABEL.search(first):
                errors.append(f"docs/author-answers.md: A8 Q{q}「结论」第一行缺少类别标签")


def headings(text):
    return [m.group(2).strip() for m in re.finditer(r"^(#{2,3})\s+(.+)$", text, re.M)]


def main():
    errors = []
    for path in md_files():
        if not path.exists():
            continue
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for no, line in enumerate(text.splitlines(), 1):
            # softmax is a standard operator, not the legacy project identifier.
            brand_line = re.sub(r"\b(?:log_)?softmax\b", "", line, flags=re.I)
            if OLD in brand_line.lower():
                allowed = CREDIT_FILE.search(path.name) and not CREDIT_IDS.sub("", line).lower().count(OLD)
                if not allowed:
                    errors.append(f"{rel}:{no}: A1 旧方法名")
            for rule, pat in SECRET_RULES.items():
                if pat.search(line):
                    errors.append(f"{rel}:{no}: A2 {rule}")
        if not rel.startswith("chapters/"):
            continue
        num = path.name[:2]
        hs = headings(text)
        if num.isdigit() and 1 <= int(num) <= 15:
            check_figures(rel, text, errors)
        if num != "00" and not CREDIT_FILE.search(path.name):
            if "本章出处" not in hs:
                errors.append(f"{rel}: A3 缺少小节「本章出处」")
            else:
                tail = text.split("本章出处", 1)[1]
                if not SOURCE_REF.search(tail):
                    errors.append(f"{rel}: A4 「本章出处」里没有源仓库路径或提交号")
        if num in TOPIC_SECTIONS:
            for sec in TOPIC_SECTIONS[num]:
                if not any(h == sec or h.endswith(sec) for h in hs):
                    errors.append(f"{rel}: A3 专题章节缺少小节「{sec}」")
            if not PITFALL_HEADER.search(text) and num == "06":
                errors.append(f"{rel}: A3 「坑与修复」需要表头 | 现象 | 根因 | 修复 | 证据 |")
            if num == "13" and not SIGNAL_HEADER.search(text):
                errors.append(f"{rel}: A3 「信号总表」需要表头 | 信号 | 公式 | 样本量 | 数据来源 | 原阈值 | 修订阈值 | 为什么这样算 | 证据 |")
        if num in STAGE_CHAPTERS:
            for sec in STAGE_SECTIONS:
                if not any(h == sec or h.endswith(sec) for h in hs):
                    errors.append(f"{rel}: A3 训练阶段章节缺少小节「{sec}」")
            if not PITFALL_HEADER.search(text):
                errors.append(f"{rel}: A3 「坑与修复」需要表头 | 现象 | 根因 | 修复 | 证据 |")
    examples = ROOT / "examples"
    if examples.is_dir():
        for ex in sorted(examples.rglob("*.py")):
            if ex.name.startswith(("test_", "_")):
                continue
            if not list(ROOT.glob(f"tests/**/test_{ex.stem}.py")):
                errors.append(f"{ex.relative_to(ROOT).as_posix()}: A5 缺少测试 tests/test_{ex.stem}.py")
    plots = ROOT / "assets" / "images" / "plots"
    if plots.is_dir():
        for img in sorted(plots.iterdir()):
            if img.suffix.lower() not in (".png", ".svg", ".jpg", ".jpeg"):
                continue
            script = ROOT / "scripts" / "figures" / f"{img.stem}.py"
            if not script.exists():
                errors.append(f"assets/images/plots/{img.name}: A7 缺少画图脚本 scripts/figures/{img.stem}.py")
            elif "DATA_SOURCE" not in script.read_text(encoding="utf-8"):
                errors.append(f"scripts/figures/{script.name}: A7 没有声明 DATA_SOURCE")
    check_answers(errors)
    if errors:
        print("\n".join(errors))
        sys.exit(f"check_book: {len(errors)} 处不通过")
    print("check_book: OK")


if __name__ == "__main__":
    main()
