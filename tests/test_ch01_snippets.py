import ast
from pathlib import Path
import re
import runpy
import textwrap

ROOT = Path(__file__).resolve().parents[1]

def test_chapter_snippets_match_and_run():
    module_path = ROOT/'examples/ch01_snippets.py'
    source = module_path.read_text()
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    blocks = re.findall(r'```python\n(.*?)```', (ROOT/'chapters/01_pytorch.md').read_text(), re.S)
    assert len(blocks) == len(functions) == 6
    lines = source.splitlines(keepends=True)
    for block, node in zip(blocks, functions):
        # Include comments: end_lineno excludes trailing comments, so slice to next function.
        start = node.lineno
        next_starts = [f.lineno-1 for f in functions if f.lineno > node.lineno]
        end = min(next_starts) if next_starts else source[:source.index('if __name__')].count('\n')
        actual = textwrap.dedent(''.join(lines[start:end])).strip()
        assert actual == block.strip()
    module = runpy.run_path(str(module_path))
    for i in range(1, 7):
        module[f'snippet_{i}']()
