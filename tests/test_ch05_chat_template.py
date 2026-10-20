import inspect
from pathlib import Path
import re
import textwrap

import pytest
from examples.ch05_chat_template import (
    END, PREFIX, START, chapter_snippet, demo, full_render,
    generated_turn, observation_suffix, parse_one_action, tool_call,
)


@pytest.mark.parametrize("observations", [["a"], ["a", "b"], [" line 1\nline 2 "]])
def test_incremental_and_full_text(observations):
    turns = [("虚构推理", "printf 'a\nb'", observations), ("", "pwd", [])]
    incremental = START + "user\n任务" + END + "\n" + PREFIX
    incremental += generated_turn(*turns[0][:2]) + observation_suffix(observations)
    incremental += generated_turn(*turns[1][:2])
    assert full_render("任务", turns) == incremental + "\n"
    assert full_render("任务", turns) != incremental


def test_reasoning_is_not_an_action():
    hidden = tool_call("printf 'not executed'")
    text = hidden + "\n</think>\n\n" + tool_call("pwd") + END
    assert parse_one_action(text) == "pwd"
    with pytest.raises(ValueError):
        parse_one_action(hidden)


@pytest.mark.parametrize("text", [
    "unclosed thinking", "r</think>no call", "r</think>" + tool_call("pwd") * 2,
    "r</think>" + tool_call("pwd").replace("function=bash", "function=python"),
    "r</think>" + tool_call("pwd") + "untrusted tail",
])
def test_invalid_action_contract(text):
    with pytest.raises(ValueError):
        parse_one_action(text)


@pytest.mark.parametrize("command", ["", " ", 4, "echo </parameter>"])
def test_invalid_command(command):
    with pytest.raises(ValueError):
        tool_call(command)


def test_no_observation_is_not_an_empty_tool_turn():
    with pytest.raises(ValueError):
        observation_suffix([])
    demo()


def test_chapter_snippet_matches():
    chapter = (Path(__file__).resolve().parents[1] / "chapters/05_chat_template.md").read_text()
    expected = textwrap.dedent("\n".join(inspect.getsource(chapter_snippet).splitlines()[1:])).strip()
    assert re.findall(r"^```python\n(.*?)^```", chapter, re.M | re.S) == [expected + "\n"]
    chapter_snippet()
