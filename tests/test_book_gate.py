"""Regression checks for the upstream review gate; do not weaken branding checks."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('book_gate', ROOT/'scripts/check_book.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

@pytest.mark.parametrize('line,allowed', [
    ('softmax 和 log_softmax 是算子', True),
    ('TMAX 是旧标识', False),
    ('softmax 与 tmax 不是同一个词', False),
])
def test_branding_does_not_reject_softmax(tmp_path, monkeypatch, line, allowed):
    chapter = tmp_path/'chapters'/'00_preface.md'
    chapter.parent.mkdir()
    chapter.write_text('# 前言\n'+line, encoding='utf-8')
    monkeypatch.setattr(gate, 'ROOT', tmp_path)
    if allowed:
        gate.main()
    else:
        with pytest.raises(SystemExit):
            gate.main()
