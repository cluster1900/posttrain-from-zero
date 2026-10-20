"""Validate sanitized chart data, local rendering, and illustration references."""
import ast
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA_SOURCE = 'result/round10/conclusion.md 最终结项总结 §2.1（提交 ca57456）'


def test_recorded_chart_percentages():
    data = json.loads((ROOT/'data/ch00_metrics.json').read_text())
    assert data['source_commit'] == 'ca57456'
    assert data['source'] == 'result/round10/conclusion.md'
    assert data['tasks'] * data['attempts_per_task'] == data['trials_per_stage'] == 445
    keys = ['pass_at_1', 'pass_at_3', 'pass_at_5', 'valid_tool_call_rate']
    assert [[stage[k] for k in keys] for stage in data['stages']] == [
        ['0.00', '0.00', '0.00', '88.96'],
        ['1.12', '3.03', '4.49', '96.58'],
        ['0.90', '2.36', '3.37', '96.24'],
    ]
    readme = (ROOT/'data/README.md').read_text()
    assert data['source'] in readme and data['source_commit'] in readme
    assert not (ROOT/'scripts/data/batch1_metrics.json').exists()


def test_chinese_figure_references_and_xml():
    refs = [ref for p in [(ROOT/'chapters/00_preface.md'), (ROOT/'chapters/01_pytorch.md')]
            for ref in re.findall(r'!\[[^\]]*\]\((images/[^)]+)\)', p.read_text())]
    assert len(refs) == len(set(refs)) == 5
    for filename, expected in [('00_preface.md', ['0-1', '0-2']),
                               ('01_pytorch.md', ['1-1', '1-2', '1-3'])]:
        captions = re.findall(r'!\[图 (\d-\d)', (ROOT/'chapters'/filename).read_text())
        assert captions == expected
    svgs = [ref for ref in refs if ref.endswith('.svg')]
    assert len(svgs) == 4
    for ref in svgs:
        text = (ROOT/'assets'/ref).read_text()
        ET.fromstring(text)
        assert 'Noto Sans CJK SC' in text
        assert re.search(r'[\u4e00-\u9fff]', text)
    assert 'images/plots/ch00_metrics.png' in refs
    png = (ROOT/'assets/images/plots/ch00_metrics.png').read_bytes()
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    assert struct.unpack('>II', png[16:24]) == (1920, 1088)
    assert not (ROOT/'assets/images/ch00_metrics.svg').exists()


def test_chart_generator_source_and_out(tmp_path):
    script = ROOT/'scripts/figures/ch00_metrics.py'
    module = ast.parse(script.read_text())
    values = {target.id: ast.literal_eval(node.value)
              for node in module.body if isinstance(node, ast.Assign)
              for target in node.targets if isinstance(target, ast.Name) and target.id == 'DATA_SOURCE'}
    assert values['DATA_SOURCE'] == DATA_SOURCE
    out = tmp_path/'rendered-plots'
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path/'matplotlib'),
               XDG_CACHE_HOME=str(tmp_path/'cache'))
    result = subprocess.run([sys.executable, str(script), '--out', str(out)],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not re.search(r'Glyph .* missing', result.stderr)
    png = (out/'ch00_metrics.png').read_bytes()
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    assert DATA_SOURCE.encode() in png
    assert struct.unpack('>II', png[16:24]) == (1920, 1088)
    # Same runtime, input and font must regenerate identical bytes.
    again = subprocess.run([sys.executable, str(script), '--out', str(out)],
                           cwd=tmp_path, env=env, capture_output=True, text=True)
    assert again.returncode == 0, again.stderr
    assert (out/'ch00_metrics.png').read_bytes() == png
