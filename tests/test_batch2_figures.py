import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_retention_source_and_plot(tmp_path):
    data = json.loads((ROOT / 'data/ch04_retention.json').read_text())
    assert data['source_commit'] == 'ca57456'
    assert [(p['max_seq_len'], p['retained']) for p in data['points']] == [
        (16384, 5330), (24576, 5618), (32768, 5711)
    ]
    assert data['total_trajectories'] - data['points'][1]['retained'] == 177
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'),
               XDG_CACHE_HOME=str(tmp_path / 'cache'))
    script = ROOT / 'scripts/figures/ch04_retention.py'
    subprocess.run([sys.executable, str(script), '--out', str(tmp_path)],
                   env=env, check=True, cwd=tmp_path)
    assert (tmp_path / 'ch04_retention.png').read_bytes() == (
        ROOT / 'assets/images/plots/ch04_retention.png'
    ).read_bytes()
