import hashlib
import inspect
import re
import textwrap
from pathlib import Path
import pytest
from examples.ch04_data_freeze import chapter_snippet, split_tasks, freeze, demo


def records():
    return [{'task_id': f'toy-{i}', 'instruction': 'synthetic', 'has_test': True} for i in range(12)]


def test_split_and_hash(tmp_path):
    held, train = split_tasks(records(), heldout_size=2)
    assert len(held) == 2 and len(train) == 10
    assert {r['task_id'] for r in held}.isdisjoint(r['task_id'] for r in train)
    assert split_tasks(list(reversed(records())), heldout_size=2) == (held, train)
    a = freeze(records(), tmp_path, heldout_size=2)
    assert a == freeze(records()[::-1], tmp_path, heldout_size=2)
    p = tmp_path/'train.jsonl'
    p.write_bytes(p.read_bytes()+b'\n')
    assert hashlib.sha256(p.read_bytes()).hexdigest() != a['train.jsonl']


@pytest.mark.parametrize('size', [-1, 12, True])
def test_invalid_sizes(size):
    with pytest.raises(ValueError):
        split_tasks(records(), heldout_size=size)


def test_fail_closed(tmp_path):
    freeze(records(), tmp_path)
    bad = records(); bad[0]['has_test'] = False
    with pytest.raises(ValueError):
        freeze(bad, tmp_path)
    assert not list(tmp_path.glob('*.jsonl'))
    with pytest.raises(ValueError):
        split_tasks(records()+records()[:1])
    demo()


def test_chapter_snippet():
    text = (Path(__file__).resolve().parents[1]/'chapters/04_data_freeze.md').read_text()
    blocks = re.findall(r'^```python\n(.*?)^```', text, re.M | re.S)
    source = textwrap.dedent('\n'.join(inspect.getsource(chapter_snippet).splitlines()[1:])).strip()
    assert blocks == [source+'\n']
    chapter_snippet()


def test_recorded_duration():
    from datetime import datetime
    start=datetime.fromisoformat('2026-10-12T05:11:42+00:00')
    end=datetime.fromisoformat('2026-10-12T05:31:45+00:00')
    assert (end-start).total_seconds()==1203


@pytest.mark.parametrize('bad_row', [
    {'instruction': 'synthetic', 'has_test': True},
    {'task_id': [], 'instruction': 'synthetic', 'has_test': True},
    {'task_id': '', 'instruction': 'synthetic', 'has_test': True},
    {'task_id': 'toy', 'instruction': 42, 'has_test': True},
    {'task_id': 'toy', 'instruction': 'synthetic', 'has_test': 'yes'},
    {'task_id': 'toy', 'instruction': 'synthetic', 'has_test': 1},
    {'task_id': 'toy', 'instruction': 'synthetic', 'has_test': True, 'extra': object()},
    None,
])
def test_malformed_records_remove_old_manifests(tmp_path, bad_row):
    freeze(records(), tmp_path)
    invalid = records()
    invalid[0] = bad_row
    with pytest.raises(ValueError):
        freeze(invalid, tmp_path)
    assert not list(tmp_path.glob('*.jsonl'))


@pytest.mark.parametrize('invalid,seed', [(None, 42), ({}, 42), ([], True)])
def test_invalid_container_or_seed_removes_old_manifests(tmp_path, invalid, seed):
    freeze(records(), tmp_path)
    with pytest.raises(ValueError):
        freeze(invalid, tmp_path, seed=seed)
    assert not list(tmp_path.glob('*.jsonl'))
