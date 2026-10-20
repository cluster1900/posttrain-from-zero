"""CPU 教学断言的 pytest 入口；网络调用会使测试失败。"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import socket
import subprocess
import sys

import pytest
import torch

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "ch01_pytorch.py"
spec = importlib.util.spec_from_file_location("ch01_pytorch", EXAMPLE_PATH)
assert spec is not None and spec.loader is not None
examples = importlib.util.module_from_spec(spec)
spec.loader.exec_module(examples)


@pytest.fixture(autouse=True)
def offline_cpu(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("教学实验不能发起网络连接")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    random.seed(0)


@pytest.mark.parametrize("section", list(examples.EXAMPLES))
def test_ch01_section(section):
    examples.EXAMPLES[section]()


def test_cli_all_with_network_guard():
    # 子进程不会继承 monkeypatch，所以在独立 CLI 进程内再次安装保护。
    wrapper = """
import runpy, socket, sys

def blocked(*args, **kwargs):
    raise AssertionError('Network connect attempted during offline examples')
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
path = sys.argv[1]
sys.argv = [path, 'all']
runpy.run_path(path, run_name='__main__')
print('PASS network-connect guard')
"""
    completed = subprocess.run(
        [sys.executable, "-c", wrapper, str(EXAMPLE_PATH)],
        text=True, capture_output=True, timeout=120,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS 6 sections" in completed.stdout
    assert "PASS network-connect guard" in completed.stdout
