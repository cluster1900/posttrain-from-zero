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


def test_static_source_excerpts_never_become_teaching_blocks():
    import importlib.util
    spec = importlib.util.spec_from_file_location('built_book', ROOT/'scripts/check_built_book.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    document = '''```{.python .source-excerpt}
raise RuntimeError("must not execute this static source excerpt")
```

```python
assert 2 + 2 == 4
```
'''
    assert checker.teaching_blocks(document) == ['assert 2 + 2 == 4\n']
    chapter = (ROOT/'chapters/01_pytorch.md').read_text()
    assert chapter.count('.source-excerpt}') == 6
    assert len(checker.teaching_blocks(chapter)) == 6


def test_mixed_figure_counts_include_mermaid():
    from scripts.check_built_book import ImageCounter, expected_images
    text = '![' + '图 0-1 示意' + '](images/a.svg)\n![图 0-2 实测](images/plots/b.png)\n```{.mermaid #roadmap}\nflowchart TB\n```\n'
    refs, mermaid = expected_images([text])
    assert refs == ['images/a.svg', 'images/plots/b.png']
    assert mermaid == 1
    parser = ImageCounter()
    parser.feed('<figure><svg><title>图</title></svg></figure><img src="data:image/png;base64,x"><img src="roadmap.png">')
    assert parser.count == len(refs) + mermaid == 3
