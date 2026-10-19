"""Regression checks for the upstream review gate; do not weaken branding checks."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('book_gate', ROOT/'scripts/check_book.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
OLD = 'T' + 'MAX'

@pytest.mark.parametrize('line,allowed', [
    ('softmax 和 log_softmax 是算子', True),
    (OLD + ' 是旧标识', False),
    ('softmax 与 ' + OLD.lower() + ' 不是同一个词', False),
    (OLD + '_ROOT', False),
    ('/workspace/' + OLD.lower() + '/', False),
    ('`' + OLD.lower() + '-xxxx`', False),
])
def test_branding_standard_operator_exception(tmp_path, monkeypatch, capsys, line, allowed):
    chapter = tmp_path/'chapters'/'00_preface.md'
    chapter.parent.mkdir()
    chapter.write_text('# 前言\n'+line, encoding='utf-8')
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    if allowed:
        gate.main()
        assert 'A1' not in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit):
            gate.main()
        assert 'A1' in capsys.readouterr().out
