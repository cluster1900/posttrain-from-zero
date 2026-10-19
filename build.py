#!/usr/bin/env python3
"""build.py：把 chapters/ 下的 Markdown 编译成网页（dist/index.html）和 EPUB（dist/posttrain-from-zero.epub）。

用法：python3 build.py
依赖：pandoc ≥ 3.0（apt install pandoc / brew install pandoc）
"""
import os
import shutil
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
CHAPTERS = os.path.join(BASE, "chapters")
ASSETS = os.path.join(BASE, "assets")
DIST = os.path.join(BASE, "dist")
TITLE = "后训练从 0 到 1"
AUTHOR = "Hawk Wu"
MD_FORMAT = "markdown-blank_before_blockquote+lists_without_preceding_blankline"


def chapters():
    files = sorted(f for f in os.listdir(CHAPTERS) if f.endswith(".md"))
    if not files:
        sys.exit("chapters/ 里没有 .md 文件")
    return [os.path.join(CHAPTERS, f) for f in files]


def run(args):
    print("+", " ".join(args))
    subprocess.run(args, check=True, cwd=BASE)


def main():
    if shutil.which("pandoc") is None:
        sys.exit("找不到 pandoc：apt install pandoc 或 brew install pandoc")
    if os.path.isdir(DIST):
        shutil.rmtree(DIST)
    os.makedirs(DIST)
    if os.path.isdir(os.path.join(ASSETS, "images")):
        shutil.copytree(os.path.join(ASSETS, "images"), os.path.join(DIST, "images"))
    common = ["-f", MD_FORMAT, "--toc", "--toc-depth=2",
              "--metadata", f"title={TITLE}", "--metadata", f"author={AUTHOR}",
              "--metadata", "lang=zh-CN", "--resource-path", f"{CHAPTERS}:{ASSETS}"]
    src = chapters()
    run(["pandoc", *src, *common, "-s", "--mathml", "--embed-resources",
         "--css", os.path.join(ASSETS, "style.css"), "-o", os.path.join(DIST, "index.html")])
    run(["pandoc", *src, *common, "--mathml", "--split-level=1",
         "--css", os.path.join(ASSETS, "style.css"), "-o", os.path.join(DIST, "posttrain-from-zero.epub")])
    open(os.path.join(DIST, ".nojekyll"), "w").close()
    print("OK:", ", ".join(sorted(os.listdir(DIST))))


if __name__ == "__main__":
    main()
