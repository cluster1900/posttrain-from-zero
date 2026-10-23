"""Exact source values and output size for the sanitized historical chart."""
import json
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_builds_values_and_reproduction(tmp_path):
    data = json.loads((ROOT/'data/ch08_builds.json').read_text())
    assert data['source_commit'] == 'ca57456'
    assert data['source'] == 'result/round8/costs-and-retention.json'
    assert data['billing_amount'] is None
    assert [(x['status'],x['observed_wall_seconds']) for x in data['builds']] == [
        ('FAILURE',255.836),('CANCELLED',1590.938),('SUCCESS',1579.385)]
    assert round(sum(x['observed_wall_seconds'] for x in data['builds']),3) == data['total_observed_build_wall_seconds'] == 3426.159
    subprocess.run([sys.executable,str(ROOT/'scripts/figures/ch08_builds.py'),'--out',str(tmp_path)],check=True)
    generated = (tmp_path/'ch08_builds.png').read_bytes()
    assert struct.unpack('>II',generated[16:24]) == (1920,1088)
    assert generated == (ROOT/'assets/images/plots/ch08_builds.png').read_bytes()
    assert set(data) == {'source','source_commit','unit','billing_amount','builds','total_observed_build_wall_seconds'}
    assert all(set(row) == {'attempt','status','observed_wall_seconds'} for row in data['builds'])
