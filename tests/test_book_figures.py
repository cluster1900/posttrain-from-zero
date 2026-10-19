"""Validate the sanitized chart data and illustration references, not private project files."""
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]

def test_recorded_chart_percentages():
    data = json.loads((ROOT/'scripts/data/batch1_metrics.json').read_text())
    assert data['source_commit'] == 'ca57456'
    assert data['source'] == 'result/round10/conclusion.md'
    assert data['tasks'] * data['attempts_per_task'] == data['trials_per_stage'] == 445
    keys = ['pass_at_1', 'pass_at_3', 'pass_at_5', 'valid_tool_call_rate']
    assert [[stage[k] for k in keys] for stage in data['stages']] == [
        ['0.00', '0.00', '0.00', '88.96'],
        ['1.12', '3.03', '4.49', '96.58'],
        ['0.90', '2.36', '3.37', '96.24'],
    ]

def test_chinese_figure_references_and_xml():
    refs = [ref for p in (ROOT/'chapters').glob('*.md')
            for ref in re.findall(r'!\[[^\]]*\]\((images/[^)]+)\)', p.read_text())]
    assert len(refs) == len(set(refs)) == 5
    for ref in refs:
        text = (ROOT/'assets'/ref).read_text()
        ET.fromstring(text)
        assert 'Noto Sans CJK SC' in text
        assert re.search(r'[\u4e00-\u9fff]', text)
