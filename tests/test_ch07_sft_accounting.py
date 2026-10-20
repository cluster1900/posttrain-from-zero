import inspect
import json
from pathlib import Path
import re
import textwrap

from examples.ch07_sft_accounting import (
    accumulated_gradient, chapter_snippet, complete_checkpoint, optimizer_resume,
)

ROOT = Path(__file__).resolve().parents[1]


def test_token_mean_gradient_and_resumed_state():
    accumulated_gradient()
    optimizer_resume()


def test_only_complete_checkpoints_are_candidates(tmp_path):
    assert complete_checkpoint(tmp_path) is None
    for name in ("step_2", "step_10", "step_99", "step_invalid"):
        (tmp_path / name).mkdir()
    for name in ("step_2", "step_10", "step_invalid"):
        (tmp_path / name / ".complete").write_text("demo")
    assert complete_checkpoint(tmp_path).name == "step_10"


def test_chapter_snippet_matches():
    chapter = (ROOT / "chapters/07_sft.md").read_text()
    expected = textwrap.dedent("\n".join(inspect.getsource(chapter_snippet).splitlines()[1:])).strip()
    assert re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S) == [expected + "\n"]
    chapter_snippet()


def test_archived_metrics_are_complete_not_smoothed():
    data = json.loads((ROOT / "data/ch07_sft_metrics.json").read_text())
    for size, steps, memory, last_loss in (("2b", 702, 13.91, 0.2513), ("4b", 688, 21.64, 0.1431)):
        rows = data["runs"][size]
        assert [r["step"] for r in rows] == list(range(1, steps + 1))
        assert max(r["peak_mem_gb"] for r in rows) == memory
        assert round(rows[-1]["loss"], 4) == last_loss
        assert {r["epoch"] for r in rows} == {0, 1}
        assert all(r["label_tokens"] > 0 for r in rows)
