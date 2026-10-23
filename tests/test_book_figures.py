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


def test_all_chapter_figure_references_and_numbering():
    all_refs = []
    for chapter in sorted((ROOT / 'chapters').glob('*.md')):
        number = int(chapter.name.split('_')[0])
        if number == 99:
            continue
        text = chapter.read_text()
        matches = re.findall(r'!\[图 (\d+-\d+)[^\]]*\]\((images/[^)]+)\)', text)
        # Include the upstream Mermaid caption in source order, not a hardcoded total.
        numbered = re.findall(r'(?:!\[图 |caption="图 )(\d+-\d+)', text)
        minimum = 4 if number in (6, 8, 9, 10) else 2
        assert len(numbered) >= minimum, chapter.name
        assert numbered == [f'{number}-{i + 1}' for i in range(len(numbered))], chapter.name
        for _, ref in matches:
            path = ROOT / 'assets' / ref
            assert path.is_file(), ref
            all_refs.append(ref)
            if path.suffix == '.svg':
                content = path.read_text()
                ET.fromstring(content)
                assert 'Noto Sans CJK SC' in content
                assert re.search(r'[\u4e00-\u9fff]', content)
            else:
                png = path.read_bytes()
                assert png[:8] == b'\x89PNG\r\n\x1a\n'
                width, height = struct.unpack('>II', png[16:24])
                expected_sizes = {
                    'ch00_metrics.png': (1920, 1088), 'ch04_retention.png': (1600, 864),
                    'ch06_parameters.png': (1760, 912), 'ch07_sft_curves.png': (1920, 1280),
                    'ch07_memory.png': (1600, 880), 'ch08_builds.png': (1920, 1088),
                }
                assert (width, height) == expected_sizes[path.name], path.name
    assert len(all_refs) == len(set(all_refs))
    assert not (ROOT / 'assets/images/ch00_metrics.svg').exists()


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
