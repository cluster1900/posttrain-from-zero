import json
from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_ch09_evidence_has_real_curves_and_source():
    obj=json.loads((ROOT/"data/ch09_rl_2b_24k_metrics.json").read_text())
    rows=obj["records"]
    assert len(rows)==3
    assert obj["source"]["commit"]=="ca57456"
    assert "DATA_SOURCE" in (ROOT/"scripts/figures/ch09_rl_2b_24k.py").read_text()
    for row in rows:
        assert 0<=row["reward_mean"]<=1
        assert 0<=row["kept_frac"]<=1
        assert row["useful_groups"]<=row["groups"]
        assert row["gen_s"]>=0 and row["train_s"]>=0 and row["step_s"]>=0
        assert row["sync_logprob_abs_diff"]>=0

def test_ch09_numbers_are_bound_to_data(tmp_path):
    data=json.loads((ROOT/"data/ch09_rl_2b_24k_metrics.json").read_text())
    chapter=(ROOT/"chapters/09_dppo.md").read_text()
    for row in data["records"]:
        assert str(row["step_s"]) in chapter
        assert f"{row['sync_logprob_abs_diff']:.10f}" in chapter
    mutated=chapter.replace("343.6","999.9",1)
    assert "999.9" not in "\n".join(str(row["step_s"]) for row in data["records"])
    assert mutated!=chapter
