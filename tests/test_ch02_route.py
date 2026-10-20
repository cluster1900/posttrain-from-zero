from pathlib import Path
import inspect
import re
import textwrap
import pytest
from examples.ch02_route import chapter_snippet, group_signal, demo


def test_group_signals():
    demo()
    assert group_signal([0, 1], [False, False], [False, False]) == [(0, -.5, True), (1, .5, True)]
    assert group_signal([0, 0], [False, False], [False, False]) is None


@pytest.mark.parametrize('args', [([0], [], [False]), ([.5, 1], [False, False], [False, False])])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        group_signal(*args)


def test_chapter_snippet():
    text = (Path(__file__).resolve().parents[1]/'chapters/02_route.md').read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    source = textwrap.dedent('\n'.join(inspect.getsource(chapter_snippet).splitlines()[1:])).strip()
    assert blocks == [source+'\n']
    chapter_snippet()
