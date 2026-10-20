from pathlib import Path

def test_author_answers_have_a8_sections_and_source_labels():
    text=(Path(__file__).parents[1]/"docs/author-answers.md").read_text()
    assert "[记录事实]" in text and "[读者运行所得]" in text and "[作者分析]" in text
    assert "qwen-posttrain@ca57456" in text
    assert "私有 URL、账号、主机名、云项目或凭据" in text
    for q in range(1,11):
        start=text.index(f"## Q{q}：")
        end=text.find("\n## Q",start+1)
        chunk=text[start:] if end<0 else text[start:end]
        for heading in ("### 记录依据","### 常见做法与出处","### 结论"):
            assert heading in chunk

def test_q9_window_numbers_are_bound_to_text():
    text=(Path(__file__).parents[1]/"docs/author-answers.md").read_text()
    for value in ("0.0625","0.125","0.20","0.0416667","0.0822917"):
        assert value in text
    mutated=text.replace("0.0822917","0.9999999",1)
    assert mutated!=text
    assert "0.0822917" not in mutated.split("### 结论",1)[0].split("## Q9",1)[-1]
