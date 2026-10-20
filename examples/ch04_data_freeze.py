"""Small invented records demonstrate freeze invariants; no downloads or real tasks."""
import hashlib
import json
from pathlib import Path
import tempfile


def chapter_snippet():
    import hashlib
    task_ids = ['toy-a', 'toy-b', 'toy-c', 'toy-d']
    ordered = sorted(task_ids, key=lambda t: hashlib.sha256(f'42:{t}'.encode()).hexdigest())
    heldout, train = set(ordered[:1]), set(ordered[1:])
    assert not heldout & train
    assert heldout | train == set(task_ids)


def split_tasks(records, seed=42, heldout_size=1):
    # This toy schema is deliberately closed; unrelated fields cannot hide
    # unserializable values or turn malformed inputs into incidental KeyErrors.
    if type(records) is not list or type(seed) is not int:
        raise ValueError('records must be a list and seed must be an integer')
    for row in records:
        if type(row) is not dict or set(row) != {'task_id', 'instruction', 'has_test'}:
            raise ValueError('record must contain exactly the teaching schema fields')
        if any(type(row[key]) is not str or not row[key].strip()
               for key in ('task_id', 'instruction')):
            raise ValueError('task_id and instruction must be nonempty strings')
        if row['has_test'] is not True:
            raise ValueError('has_test must be the boolean True')
    ids = [row['task_id'] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate task IDs')
    if type(heldout_size) is not int or not 0 <= heldout_size < len(ids):
        raise ValueError('heldout must leave nonempty train')
    ordered = sorted(records, key=lambda r: hashlib.sha256(f"{seed}:{r['task_id']}".encode()).hexdigest())
    return ordered[:heldout_size], ordered[heldout_size:]


def encode_manifest(records):
    # Teaching format, intentionally not asserted equal to the private manifest bytes.
    return ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n'
                   for r in sorted(records, key=lambda r: r['task_id'])).encode()


def freeze(records, destination, seed=42, heldout_size=1):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = [destination/'train.jsonl', destination/'heldout.jsonl']
    try:
        heldout, train = split_tasks(records, seed, heldout_size)
    except ValueError:
        for p in paths:
            p.unlink(missing_ok=True)
        raise
    hashes = {}
    for path, rows in zip(paths, (train, heldout)):
        data = encode_manifest(rows)
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(data)
        tmp.replace(path)
        hashes[path.name] = hashlib.sha256(data).hexdigest()
    return hashes


def demo():
    chapter_snippet()
    records = [{'task_id': f'toy-{i}', 'instruction': 'Invented task', 'has_test': True} for i in range(8)]
    with tempfile.TemporaryDirectory() as d:
        first = freeze(records, d)
        assert first == freeze(list(reversed(records)), d)
        p = Path(d)/'train.jsonl'
        p.write_bytes(p.read_bytes()+b' ')
        assert hashlib.sha256(p.read_bytes()).hexdigest() != first['train.jsonl']
        broken = [dict(r) for r in records]
        broken[0]['has_test'] = False
        try:
            freeze(broken, d)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input must fail')
        assert not list(Path(d).glob('*.jsonl'))
    print('PASS: deterministic split, byte hashes, fail-closed; synthetic teaching data only')


if __name__ == '__main__':
    demo()
