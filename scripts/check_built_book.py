#!/usr/bin/env python3
"""Check built structure/assets and execute every Python chapter fence on CPU."""
from pathlib import Path
import re
import subprocess
import sys
import zipfile
from xml.etree import ElementTree as ET
root = Path(__file__).resolve().parents[1]
html = (root/'dist/index.html').read_text()
assert 'Hawk Wu' in html and '第 1 章' in html
assert html.count('<svg') + html.count('data:image/svg+xml') == 2, 'HTML must embed both figures'
with zipfile.ZipFile(root/'dist/posttrain-from-zero.epub') as z:
    assert z.read('mimetype') == b'application/epub+zip'
    for n in z.namelist():
        if n.endswith(('.xhtml', '.opf', '.svg')):
            ET.fromstring(z.read(n))
    assert len([n for n in z.namelist() if n.endswith('.svg')]) == 2
    package = z.read('EPUB/content.opf').decode()
    assert 'Hawk Wu' in package and 'zh-CN' in package
count = 0
for chapter in sorted((root/'chapters').glob('*.md')):
    text = chapter.read_text()
    assert not re.search(r'https?://[^\s)]*qwen-posttrain', text), 'private repository link'
    assert not re.search(r'gs://|AIza[\w-]{20,}|-----BEGIN .*PRIVATE KEY', text)
    for block in re.findall(r'```python\n(.*?)```', text, flags=re.S):
        subprocess.run([sys.executable, '-c', block], check=True, cwd=root)
        count += 1
print(f'PASS HTML/EPUB structure, 2 embedded figures, {count} Python code blocks')
