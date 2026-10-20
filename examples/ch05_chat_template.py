"""Original text-level teaching contract; not a replacement Qwen tokenizer/parser.

Strings are invented. No command is executed. UTF-8 equality does not prove
that independently tokenized fragments concatenate to the same token IDs.
"""
import re

START = "<|im_start|>"
END = "<|im_end|>"
PREFIX = START + "assistant\n<think>\n"


def tool_call(command):
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a nonempty string")
    if any(tag in command for tag in ("<tool_call>", "</parameter>", "</function>")):
        raise ValueError("toy grammar excludes delimiter injection")
    return ("<tool_call>\n<function=bash>\n<parameter=command>\n" + command
            + "\n</parameter>\n</function>\n</tool_call>")


def generated_turn(reasoning, command):
    return reasoning + "\n</think>\n\n" + tool_call(command) + END


def observation_suffix(observations):
    if not observations or any(not isinstance(o, str) for o in observations):
        raise ValueError("need one or more textual observations")
    blocks = "".join("\n<tool_response>\n" + o.strip() + "\n</tool_response>" for o in observations)
    return "\n" + START + "user" + blocks + END + "\n" + PREFIX


def full_render(task, turns):
    text = START + "user\n" + task.strip() + END + "\n"
    for reasoning, command, observations in turns:
        text += PREFIX + generated_turn(reasoning, command) + "\n"
        if observations:
            text += START + "user" + "".join(
                "\n<tool_response>\n" + o.strip() + "\n</tool_response>" for o in observations
            ) + END + "\n"
    return text


def parse_one_action(text):
    """Strict toy grammar: parse only the action after a closed thinking segment."""
    if "</think>" not in text:
        raise ValueError("thinking was not closed")
    body = text.split("</think>", 1)[1].strip()
    if body.endswith(END):
        body = body[:-len(END)].rstrip()
    match = re.fullmatch(
        r"<tool_call>\s*<function=bash>\s*<parameter=command>\n(.*?)\n"
        r"</parameter>\s*</function>\s*</tool_call>", body, re.S,
    )
    if not match or body.count("<tool_call>") != 1:
        raise ValueError("need exactly one bash command")
    command = match.group(1)
    tool_call(command)  # Apply the same explicit delimiter restrictions.
    return command


def chapter_snippet():
    prefix = "<|im_start|>assistant\n<think>\n"
    emitted = "检查当前目录。\n</think>\n\n工具调用文本<|im_end|>"
    assembled = prefix + emitted
    assert assembled.endswith("<|im_end|>")
    assert "<|endoftext|>" not in assembled


def demo():
    turns = [("先看输出", "printf 'toy'", ["toy", "(exit_code=0)"]),
             ("完成检查", "printf 'done'", [])]
    incremental = START + "user\n" + "虚构任务" + END + "\n" + PREFIX
    for reasoning, command, observations in turns:
        emitted = generated_turn(reasoning, command)
        assert parse_one_action(emitted) == command
        incremental += emitted
        if observations:
            incremental += observation_suffix(observations)
    assert full_render("虚构任务", turns) == incremental + "\n"
    assert full_render("虚构任务", turns).encode() == (incremental + "\n").encode()
    chapter_snippet()
    print("PASS: toy text/UTF-8 framing, grouped observations, strict action parsing; no commands executed")


if __name__ == "__main__":
    demo()
