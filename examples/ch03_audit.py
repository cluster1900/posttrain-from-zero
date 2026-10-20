#!/usr/bin/env python3
"""第 3 章：离线 CPU 审计练习，不运行 RL、不生成模型权重或生产门禁报告。

源问题：docs/apexo/00_progress.md §1.2、§1.3、§6.1–§6.4（ca57456）。
本文件所有模型数值、源码字符串和 trial 都是教学构造，不能当作项目记录。
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import tempfile


DUMMY_SOURCE = '''
def train(model, optimizer):
    dummy_response = "a fixed response"
    model.save_pretrained("not-a-real-output")
'''


def audit_training_source(source: str) -> list[str]:
    """保守静态提示，不运行被审源码，也不是完整数据流或控制流分析。"""
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    dummy_literal = any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "dummy_response"
                for target in node.targets)
        and isinstance(node.value, ast.Constant)
        for node in ast.walk(tree)
    )
    findings = []
    if dummy_literal:
        findings.append("literal_dummy_response")
    if "backward" not in calls:
        findings.append("no_direct_backward_call")
    if "step" not in calls:
        findings.append("no_direct_step_call")
    return findings


def observe_toy_step(mode: str) -> dict:
    """实际测量一维 CPU 更新；skip 是合法零更新的示意，dummy 是未训练。"""
    import torch

    if mode not in {"dummy", "backward_only", "update", "skip"}:
        raise ValueError("unknown mode")
    torch.set_num_threads(1)
    weight = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    before = weight.detach().clone()
    loss = None
    grad = None
    if mode in {"backward_only", "update"}:
        optimizer.zero_grad(set_to_none=True)
        loss = (weight - 2.0).square()
        loss.backward()
        grad = float(weight.grad)
        if mode == "update":
            optimizer.step()
    after = weight.detach().clone()
    changed = not torch.equal(before, after)
    return {
        "scope": "CPU_TOY_ONLY",
        "mode": mode,
        "loss": None if loss is None else float(loss.detach()),
        "gradient": grad,
        "before": float(before),
        "after": float(after),
        "changed": changed,
        "skipped": mode == "skip",
    }


def classify_update(evidence: dict) -> str:
    """分类这份教学观测，不证明模型质量；零更新须如实标记原因。"""
    if evidence["skipped"]:
        if evidence["changed"]:
            raise ValueError("a skipped step cannot claim a parameter update")
        return "SKIPPED_NOT_UPDATED"
    if evidence["loss"] is None or evidence["gradient"] is None:
        return "NO_BACKWARD_EVIDENCE"
    if not all(math.isfinite(evidence[key]) for key in ("loss", "gradient", "before", "after")):
        return "NONFINITE_EVIDENCE"
    if not evidence["changed"]:
        return "NO_PARAMETER_CHANGE"
    return "TOY_UPDATE_OBSERVED"


def write_demo_evidence(trials: list[dict], expected_tasks: set[str], out: Path) -> int:
    """检查教学证据完整性。没有数据返回 2，不写报告；有效也不写 PASSED。

    source_kind 必须为 fixture，阻止此函数冒充生产评测器。
    不接受缺失/损坏 timing、不同任务集或非有限奖励。
    正式门禁还需要覆盖 trial 次数、质量阈值、统计检验与训练身份。
    """
    if not trials:
        return 2
    if not expected_tasks or {trial.get("task") for trial in trials} != expected_tasks:
        raise ValueError("task set mismatch")
    for trial in trials:
        if trial.get("source_kind") != "fixture":
            raise ValueError("this function accepts teaching fixtures only")
        reward = trial.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (float, int)):
            raise ValueError("missing or invalid reward")
        if not math.isfinite(reward) or reward not in (0, 1):
            raise ValueError("reward must be finite and binary")
        timing = trial.get("timing")
        if not isinstance(timing, list) or not timing:
            raise ValueError("missing or invalid timing")
        if not all(isinstance(step, dict) and isinstance(step.get("format_error"), bool)
                   for step in timing):
            raise ValueError("invalid timing step")
    report = {
        "status": "EVIDENCE_READY_DEMO_ONLY",
        "production_gate": False,
        "n_trials": len(trials),
        "tasks": sorted(expected_tasks),
    }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    print("CPU audit demonstration; no real RL training or evaluation.")
    print("static:", ", ".join(audit_training_source(DUMMY_SOURCE)))
    for mode in ("dummy", "backward_only", "update", "skip"):
        evidence = observe_toy_step(mode)
        print(f"{mode}: {classify_update(evidence)}; changed={evidence['changed']}")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "demo_evidence.json"
        code = write_demo_evidence([], {"toy_task"}, out)
        print(f"empty evaluation: exit={code}; report_exists={out.exists()}")
        fixture = {"task": "toy_task", "reward": 1, "source_kind": "fixture",
                   "timing": [{"format_error": False}]}
        write_demo_evidence([fixture], {"toy_task"}, out)
        report = json.loads(out.read_text(encoding="utf-8"))
        print("fixture evaluation:", report["status"])


if __name__ == "__main__":
    main()
