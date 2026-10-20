import json
from pathlib import Path
import re
import subprocess
import sys
import os
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]


def test_chapter_figures():
    for n in (2,3,4):
        path=next((ROOT/'chapters').glob(f'{n:02}_*.md'))
        matches=re.findall(r'!\[图 (\d+-\d+)[^\]]*\]\((images/[^)]+)\)',path.read_text())
        assert len(matches)>=2
        assert [number for number,_ in matches]==[f'{n}-{i+1}' for i in range(len(matches))]
        for _,ref in matches:
            p=ROOT/'assets'/ref
            assert p.exists()
            if p.suffix=='.svg':
                text=p.read_text();ET.fromstring(text)
                assert 'Noto Sans CJK SC' in text


def test_retention_source_and_plot(tmp_path):
    data=json.loads((ROOT/'data/ch04_retention.json').read_text())
    assert data['source_commit']=='ca57456'
    assert [(p['max_seq_len'],p['retained']) for p in data['points']]==[(16384,5330),(24576,5618),(32768,5711)]
    assert data['total_trajectories']-data['points'][1]['retained']==177
    env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    script=ROOT/'scripts/figures/ch04_retention.py'
    subprocess.run([sys.executable,str(script),'--out',str(tmp_path)],env=env,check=True,cwd=tmp_path)
    assert (tmp_path/'ch04_retention.png').read_bytes()==(ROOT/'assets/images/plots/ch04_retention.png').read_bytes()
