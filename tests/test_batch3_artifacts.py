import json
import os
from pathlib import Path
import subprocess
import sys

from examples.ch06_model_structure import inspect_model

ROOT = Path(__file__).resolve().parents[1]


def test_reader_parameter_snapshot_matches_live_model():
    _, actual = inspect_model()
    saved = json.loads((ROOT / 'data/ch06_tiny_parameters.json').read_text())
    assert actual == saved


def test_new_data_plots_reproduce(tmp_path):
    env = dict(os.environ, MPLCONFIGDIR=str(tmp_path / 'mpl'), XDG_CACHE_HOME=str(tmp_path / 'cache'))
    for stem in ('ch06_parameters', 'ch07_sft_curves', 'ch07_memory'):
        script = ROOT / 'scripts/figures' / f'{stem}.py'
        run = subprocess.run([sys.executable, str(script), '--out', str(tmp_path)],
                             env=env, capture_output=True, text=True, cwd=tmp_path)
        assert run.returncode == 0, run.stderr
        assert 'Glyph' not in run.stderr
        assert (tmp_path / f'{stem}.png').read_bytes() == (
            ROOT / 'assets/images/plots' / f'{stem}.png'
        ).read_bytes()
