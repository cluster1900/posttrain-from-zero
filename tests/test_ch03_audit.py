"""第 3 章教学审计：离线运行，并明确其不能证明真实 RL 训练。"""
import importlib.util
import json
from pathlib import Path
import socket

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ch03_audit", ROOT / "examples/ch03_audit.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network access is not allowed")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def fixture_trial():
    return {"task": "toy_task", "reward": 1, "source_kind": "fixture",
            "timing": [{"format_error": False}]}


def test_static_flags_literal_and_missing_calls():
    assert module.audit_training_source(module.DUMMY_SOURCE) == [
        "literal_dummy_response", "no_direct_backward_call", "no_direct_step_call"]


def test_static_scan_is_not_a_correctness_certificate():
    source = "def train(loss, optimizer):\n    if False:\n        loss.backward()\n        optimizer.step()\n"
    # 这段代码永不更新，静态关键调用检查却不会报警。正文必须说明这一边界。
    assert module.audit_training_source(source) == []


@pytest.mark.parametrize("mode, expected, changed", [
    ("dummy", "NO_BACKWARD_EVIDENCE", False),
    ("backward_only", "NO_PARAMETER_CHANGE", False),
    ("update", "TOY_UPDATE_OBSERVED", True),
    ("skip", "SKIPPED_NOT_UPDATED", False),
])
def test_dynamic_observations(mode, expected, changed):
    result = module.observe_toy_step(mode)
    assert result["scope"] == "CPU_TOY_ONLY"
    assert result["changed"] is changed
    assert result["before"] == 0.0
    assert module.classify_update(result) == expected
    if mode in {"backward_only", "update"}:
        assert result["loss"] == 4.0
        assert result["gradient"] == -4.0
    assert result["after"] == pytest.approx(0.4 if changed else 0.0)


def test_nonfinite_is_not_accepted():
    evidence = module.observe_toy_step("update")
    evidence["loss"] = float("nan")
    assert module.classify_update(evidence) == "NONFINITE_EVIDENCE"


def test_skip_cannot_hide_an_update():
    evidence = module.observe_toy_step("update")
    evidence["skipped"] = True
    with pytest.raises(ValueError, match="skipped"):
        module.classify_update(evidence)


def test_no_results_never_create_report(tmp_path):
    out = tmp_path / "report.json"
    assert module.write_demo_evidence([], {"toy_task"}, out) == 2
    assert not out.exists()


def test_no_results_do_not_overwrite_old_report(tmp_path):
    out = tmp_path / "report.json"
    out.write_text("OLD REPORT, NOT A NEW RESULT", encoding="utf-8")
    assert module.write_demo_evidence([], {"toy_task"}, out) == 2
    assert out.read_text(encoding="utf-8") == "OLD REPORT, NOT A NEW RESULT"


@pytest.mark.parametrize("field, value", [
    ("timing", None), ("timing", {}), ("timing", []),
    ("timing", [{"format_error": "false"}]),
    ("reward", None), ("reward", float("nan")),
    ("reward", 0.5), ("reward", True), ("source_kind", "production"),
])
def test_invalid_evidence_never_creates_report(tmp_path, field, value):
    trial = fixture_trial()
    trial[field] = value
    out = tmp_path / "report.json"
    with pytest.raises(ValueError):
        module.write_demo_evidence([trial], {"toy_task"}, out)
    assert not out.exists()


def test_task_intersection_is_not_task_equality(tmp_path):
    out = tmp_path / "report.json"
    with pytest.raises(ValueError, match="task set"):
        module.write_demo_evidence([fixture_trial()], {"toy_task", "missing_task"}, out)
    assert not out.exists()


def test_valid_fixtures_still_never_claim_production_pass(tmp_path):
    out = tmp_path / "report.json"
    assert module.write_demo_evidence([fixture_trial()], {"toy_task"}, out) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report == {"status": "EVIDENCE_READY_DEMO_ONLY", "production_gate": False,
                      "n_trials": 1, "tasks": ["toy_task"]}
    assert "PASSED" not in out.read_text(encoding="utf-8")


def test_cli_in_process(capsys):
    module.main()
    text = capsys.readouterr().out
    assert "empty evaluation: exit=2; report_exists=False" in text
    assert "fixture evaluation: EVIDENCE_READY_DEMO_ONLY" in text


def test_chapter_python_snippet_matches_example():
    chapter = (ROOT / "chapters/03_v1_to_v2.md").read_text(encoding="utf-8")
    snippet = '''    if not trials:
        return 2'''
    assert snippet in chapter
    assert snippet in (ROOT / "examples/ch03_audit.py").read_text(encoding="utf-8")
