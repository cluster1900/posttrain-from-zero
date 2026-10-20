from pathlib import Path

def test_author_answers_have_evidence_labels_and_private_source_policy():
    text = (Path(__file__).parents[1] / "docs/author-answers.md").read_text()
    assert "[记录事实]" in text
    assert "[读者运行所得]" in text
    assert "[作者分析]" in text
    assert "qwen-posttrain@ca57456" in text
    assert "私有仓库链接、账号、主机名、云项目或凭据" in text
    for q in range(1, 11):
        assert f"Q{q}：" in text
