#!/usr/bin/env python3
"""check_book.py：审核门禁里能自动检查的部分。CI 在构建前运行；本地用法：python3 scripts/check_book.py

检查项（对应 docs/outline.md「验收标准」）：
  A1 品牌：书稿中不出现旧方法名；外部数据集 / harness 的 ID 只允许出现在致谢与许可附录。
  A2 保密：书稿与大纲中不出现云项目编号、bucket、服务账户、镜像仓库地址、邮箱、Drive 文件夹 ID、私钥或 API key。
  A3 结构：每个训练阶段章节必须包含规定的小节；每章（前言除外）必须有「本章出处」。
  A4 出处：「本章出处」里至少列出一条源仓库路径或提交号。
  A5 示例：examples/ 下每个 .py 示例都要有对应测试（tests/test_<示例名>.py）。
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
STAGE_CHAPTERS = {"03", "04", "06", "07", "08", "09", "10", "11"}
STAGE_SECTIONS = ["动机", "配置", "命令", "时间线与耗时", "坑与修复", "花费", "教训", "本章出处"]
PITFALL_HEADER = re.compile(r"\|\s*现象\s*\|\s*根因\s*\|\s*修复\s*\|\s*证据\s*\|")
SOURCE_REF = re.compile(r"`[^`]*(/|\.md|\.json|\.jsonl|\.py|\.yaml|\.log)[^`]*`|提交 `[0-9a-f]{7,40}`")


def md_files():
    yield from sorted((ROOT / "chapters").glob("*.md"))
    yield ROOT / "docs" / "outline.md"
    yield ROOT / "README.md"


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
            if OLD in line.lower():
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
        if num != "00" and not CREDIT_FILE.search(path.name):
            if "本章出处" not in hs:
                errors.append(f"{rel}: A3 缺少小节「本章出处」")
            else:
                tail = text.split("本章出处", 1)[1]
                if not SOURCE_REF.search(tail):
                    errors.append(f"{rel}: A4 「本章出处」里没有源仓库路径或提交号")
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
    if errors:
        print("\n".join(errors))
        sys.exit(f"check_book: {len(errors)} 处不通过")
    print("check_book: OK")


if __name__ == "__main__":
    main()
