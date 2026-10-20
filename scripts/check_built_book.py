#!/usr/bin/env python3
"""Check built assets and execute only plain Python teaching fences, never source excerpts."""
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from xml.etree import ElementTree as ET

TEACHING_FENCE = re.compile(r'^```python\n(.*?)^```\s*$', re.M | re.S)
IMAGE_REFERENCE = re.compile(r'!\[[^\]]*\]\((images/[^)]+)\)')

def teaching_blocks(text):
    """Attribute fences such as {.python .source-excerpt} are intentionally excluded."""
    return TEACHING_FENCE.findall(text)

class ImageCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.count = 0

    def handle_starttag(self, tag, attrs):
        if tag in {'img', 'svg'}:
            self.count += 1


def expected_images(texts):
    refs = [ref for text in texts for ref in IMAGE_REFERENCE.findall(text)]
    mermaid = sum(len(re.findall(r'^```\{\.mermaid\b', text, re.M)) for text in texts)
    return refs, mermaid


def main():
    root = Path(__file__).resolve().parents[1]
    texts = [p.read_text() for p in sorted((root/'chapters').glob('*.md'))]
    refs, mermaid = expected_images(texts)
    assert refs, 'Book must reference its illustrations'
    for ref in refs:
        assert (root/'assets'/ref).is_file(), f'Missing referenced image: {ref}'
    html = (root/'dist/index.html').read_text()
    assert 'Hawk Wu' in html and '第 1 章' in html
    counter = ImageCounter()
    counter.feed(html)
    assert counter.count == len(refs) + mermaid, 'HTML image reference/embed count mismatch'
    with zipfile.ZipFile(root/'dist/posttrain-from-zero.epub') as z:
        assert z.read('mimetype') == b'application/epub+zip'
        for n in z.namelist():
            if n.endswith(('.xhtml', '.opf', '.svg')):
                ET.fromstring(z.read(n))
        unique_refs = {Path(ref).name for ref in refs}
        images = [n for n in z.namelist() if Path(n).suffix.lower() in {'.svg', '.png', '.jpg', '.jpeg'}]
        assert len(images) == len(unique_refs) + mermaid, 'EPUB image count mismatch'
        for ref in refs:
            data = (root/'assets'/ref).read_bytes()
            assert any(z.read(n) == data for n in images), f'EPUB missing exact image: {ref}'
        package = z.read('EPUB/content.opf').decode()
        assert 'Hawk Wu' in package and 'zh-CN' in package
    count = 0
    for text in texts:
        assert not re.search(r'https?://[^\s)]*qwen-posttrain', text), 'private repository link'
        assert not re.search(r'gs://|AIza[\w-]{20,}|-----BEGIN .*PRIVATE KEY', text)
        for block in teaching_blocks(text):
            subprocess.run([sys.executable, '-c', block], check=True, cwd=root)
            count += 1
    print(f'PASS HTML/EPUB structure, {len(refs) + mermaid} embedded figures, {count} teaching Python code blocks; source excerpts not executed')

if __name__ == '__main__':
    main()
