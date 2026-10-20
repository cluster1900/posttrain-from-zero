import json
from pathlib import Path

def test_ch09_evidence_has_real_curves():
    p = Path(__file__).parents[1] / "data/ch09_rl_2b_24k_metrics.json"
    obj = json.loads(p.read_text())
    rows = obj["records"]
    assert len(rows) >= 3
    for row in rows:
        assert 0 <= row["reward_mean"] <= 1
        assert 0 <= row["kept_frac"] <= 1
        assert row["useful_groups"] <= row["groups"]
        assert row["gen_s"] >= 0 and row["train_s"] >= 0 and row["step_s"] >= 0
        assert row["sync_logprob_abs_diff"] >= 0
    assert obj["source"]["commit"] == "ca57456"
    assert obj["source"]["path"].endswith("rl_2b_24k_metrics.jsonl")
