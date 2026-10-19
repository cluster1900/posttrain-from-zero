#!/usr/bin/env python3
"""《后训练从0到1》第一章：独立、CPU-only、无 Hub 下载的教学实验。

运行：python examples/ch01_pytorch.py [all|tensor|autograd|resume|lora|causal|transformers]
这些实验不是参考项目训练入口，不代表 Qwen 训练效果或硬件性能。
仅使用临时目录保存自建 checkpoint；不加载不可信 checkpoint。
"""
from __future__ import annotations

import argparse
import copy
import os
import platform
import random
import tempfile
from pathlib import Path

# 必须在导入 transformers 之前设置；只实例化本地随机模型。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def tensor_example():
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cpu")
    embedding = nn.Embedding(8, 4).cpu()
    hidden = embedding(ids)
    assert ids.dtype == torch.int64 and ids.device.type == "cpu"
    assert hidden.shape == (1, 3, 4) and hidden.dtype == torch.float32
    low = hidden.to(dtype=torch.bfloat16)
    assert low.dtype == torch.bfloat16 and hidden.dtype == torch.float32
    # dtype/device 是两个维度；to() 的返回值必须接住。
    assert low.device.type == "cpu"
    assert hidden.numel() * hidden.element_size() == 48
    assert low.numel() * low.element_size() == 24
    assert low.float().dtype == torch.float32
    assert not ids.requires_grad and hidden.requires_grad


def autograd_example():
    """不同长度微批：先对 token 求和，再除以整个更新窗口 token 数。"""
    torch.manual_seed(11)
    model = nn.Linear(3, 5).double()
    accumulated, wrong = copy.deepcopy(model), copy.deepcopy(model)
    x = torch.randn(7, 3, dtype=torch.float64)
    y = torch.tensor([0, 1, 2, 3, 4, 1, 0])
    slices = [slice(0, 2), slice(2, 7)]  # 2 和 5 个监督 token
    total = y.numel()
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1)
                  for m in (model, accumulated, wrong)]
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    F.cross_entropy(model(x), y, reduction="sum").div(total).backward()
    for part in slices:
        loss_sum = F.cross_entropy(accumulated(x[part]), y[part], reduction="sum")
        (loss_sum / total).backward()
        # 反例：均分微批均值会给短序列过高权重。
        (F.cross_entropy(wrong(x[part]), y[part]) / len(slices)).backward()
    for full, micro in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(full.grad, micro.grad, atol=1e-12, rtol=1e-12)
    assert any(not torch.allclose(a.grad, b.grad)
               for a, b in zip(model.parameters(), wrong.parameters()))
    # 只在所有微批 backward 完毕后裁剪一次，再 step 一次。
    norms = [nn.utils.clip_grad_norm_(m.parameters(), 0.05)
             for m in (model, accumulated)]
    assert norms[0] > 0.05  # 本例确实触发裁剪
    torch.testing.assert_close(norms[0], norms[1], atol=1e-12, rtol=1e-12)
    for m in (model, accumulated):
        clipped = torch.sqrt(sum(p.grad.square().sum() for p in m.parameters()))
        assert clipped <= 0.050001
    for opt in optimizers[:2]:
        opt.step()
        opt.zero_grad(set_to_none=True)
    for a, b in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        assert a.grad is None and b.grad is None
    # 等价前提：无跨样本算子、相同有效样本与随机性；不声称任意模型逐位等价。


class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.3),
                                    nn.Linear(8, 1))
        self.register_buffer("scale", torch.tensor(1.0))

    def forward(self, x):
        return self.layers(x) * self.scale


def resume_example():
    """从 optimizer-step 边界恢复：参数、buffer、Adam 状态、进度、RNG。"""
    torch.manual_seed(17)
    random.seed(17)
    original = TinyRegressor().train()
    opt = torch.optim.AdamW(original.parameters(), lr=0.01)

    def step(model, optimizer):
        # 本例的随机数据与 dropout 依赖 torch RNG；另有 Python RNG。
        x = torch.randn(4, 3) * (0.5 + random.random())
        target = x.sum(-1, keepdim=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return loss.detach().clone()

    step(original, opt)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trusted_self_created.pt"
        torch.save({"model": original.state_dict(), "optimizer": opt.state_dict(),
                    "global_step": 1, "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate()}, path)
        expected_loss = step(original, opt)
        expected = copy.deepcopy(original.state_dict())
        expected_optimizer = copy.deepcopy(opt.state_dict())
        restored = TinyRegressor().train()
        resumed_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
        state = torch.load(path, map_location="cpu", weights_only=True)
        restored.load_state_dict(state["model"], strict=True)
        resumed_opt.load_state_dict(state["optimizer"])
        assert state["global_step"] == 1 and "scale" in state["model"]
        # 初始化模型也消耗随机数，因此必须在初始化与加载之后恢复 RNG。
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        actual_loss = step(restored, resumed_opt)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
        for pid, slots in resumed_opt.state_dict()["state"].items():
            for key, value in slots.items():
                torch.testing.assert_close(value, expected_optimizer["state"][pid][key],
                                           rtol=0, atol=0)
    # 真正的训练还需 scheduler/scaler、采样器位置等；本例没有这些组件。
    # state_dict 不保存模型结构或 train/eval 模式，构造与模式要显式重建。


class ToyLoRA(nn.Module):
    def __init__(self, width=8, rank=2, alpha=4.0):
        super().__init__()
        self.base = nn.Linear(width, width, bias=False)
        self.base.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(rank, width) * 0.02)
        self.b = nn.Parameter(torch.zeros(width, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scaling


def lora_example():
    torch.manual_seed(23)
    plain = ToyLoRA()
    recomputed = copy.deepcopy(plain)
    x = torch.randn(3, 8)  # 故意 requires_grad=False，模拟冻结前层的输出
    target = torch.randn(3, 8)
    frozen_before = plain.base.weight.detach().clone()
    opt = torch.optim.SGD([p for p in plain.parameters() if p.requires_grad], lr=0.2)
    assert sum(p.numel() for p in plain.parameters() if p.requires_grad) == 32
    outputs = []
    for model, use_checkpoint in [(plain, False), (recomputed, True)]:
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = (checkpoint(model, x, use_reentrant=False)
                      if use_checkpoint else model(x))
        outputs.append(output.detach().clone())
        assert output.dtype == torch.bfloat16
        # 参数仍为 fp32；敏感的损失归约在 fp32 中进行。
        assert model.a.dtype == torch.float32
        loss = F.mse_loss(output.float(), target)
        loss.backward()
        assert model.base.weight.grad is None
        assert model.a.grad is not None and model.b.grad is not None
        assert torch.isfinite(loss) and model.b.grad.abs().sum() > 0
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    for a, b in zip(plain.parameters(), recomputed.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    # B=0 时首步 A 的梯度为 0，这是初始化的结果，不是断图。
    assert torch.count_nonzero(plain.a.grad) == 0
    opt.step()
    torch.testing.assert_close(plain.base.weight, frozen_before, rtol=0, atol=0)
    assert torch.count_nonzero(plain.b) > 0
    # 不测速度/峰值内存，不以 CPU bf16 实验推断 GPU 性能或显存收益。


def causal_example():
    torch.manual_seed(29)
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 0, 0]])
    # 这是 token 位置上的目标 mask，不是 hidden-state 位置上的 mask。
    assistant = torch.tensor([[False, False, False, True, True],
                              [False, False, True, False, False]])
    attention = ids.ne(0)
    logits = torch.randn(2, 5, 8, requires_grad=True)
    labels = ids.clone()
    labels[~(assistant & attention)] = -100
    shifted_labels = labels[:, 1:]
    shifted_logits = logits[:, :-1, :]
    valid = shifted_labels.ne(-100)
    count = valid.sum()
    assert count.item() == 3
    loss = F.cross_entropy(shifted_logits.reshape(-1, 8), shifted_labels.reshape(-1),
                           ignore_index=-100, reduction="sum") / count
    logp = shifted_logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    manual = -logp[valid].sum() / count
    torch.testing.assert_close(loss, manual)
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, -1]) == 0
    assert torch.count_nonzero(logits.grad[:, :-1][~valid]) == 0
    assert logits.grad[:, :-1][valid].abs().sum() > 0
    # 零监督 token 的窗口应跳过/拒绝，不能让 mean(empty) 变为 NaN。
    def safe_mean(loss_sum, n):
        if n == 0:
            raise ValueError("此更新窗口没有监督 token，应跳过该窗口")
        return loss_sum / n
    try:
        safe_mean(torch.tensor(0.0), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-token guard missing")


def transformers_example():
    """自建 tokenizer + 随机 GPT2，仅演示接口，不是预训练模型。"""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config,
                              GPT2LMHeadModel, PreTrainedTokenizerFast)

    words = ["<pad>", "<unk>", "<user>", "<assistant>", "<eos>", "你好", "你好呀"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                       pad_token="<pad>", eos_token="<eos>",
                                       additional_special_tokens=["<user>", "<assistant>"])
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<' + message['role'] + '> ' + "
        "message['content'] + ' <eos> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    )
    messages = [{"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好呀"}]
    # 显式指定返回 token IDs，而不是 Transformers 5 的默认 BatchEncoding。
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, return_tensors="pt",
                                        add_generation_prompt=False)
    assert ids.tolist() == [[2, 5, 4, 3, 6, 4]]
    prefix = tokenizer.apply_chat_template(messages[:1], tokenize=True, return_dict=False,
                                           add_generation_prompt=True)
    assert prefix == [2, 5, 4, 3]
    config = GPT2Config(vocab_size=len(tokenizer), n_positions=32, n_ctx=32,
                        n_embd=16, n_layer=1, n_head=2, resid_pdrop=0.0,
                        embd_pdrop=0.0, attn_pdrop=0.0, bos_token_id=None,
                        eos_token_id=4, pad_token_id=0, use_cache=False)
    model = GPT2LMHeadModel(config).cpu().train()
    labels = ids.clone()
    labels[:, :4] = -100  # 仅监督 assistant 内容和结束 token，不监督角色标记。
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
    manual = F.cross_entropy(out.logits[:, :-1].reshape(-1, len(tokenizer)),
                             labels[:, 1:].reshape(-1), ignore_index=-100)
    torch.testing.assert_close(out.loss, manual)
    out.loss.backward()
    assert model.transformer.wte.weight.grad is not None
    assert torch.isfinite(out.loss) and out.logits.shape == (1, 6, len(tokenizer))
    assert all(p.device.type == "cpu" for p in model.parameters())
    # 模型接口内部 shift；不要把已 shift 的 labels 再传入产生二次错位。
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        model.save_pretrained(tmp, safe_serialization=True)
        loaded = AutoTokenizer.from_pretrained(tmp, local_files_only=True)
        restored = AutoModelForCausalLM.from_pretrained(tmp, local_files_only=True).cpu()
        assert loaded.chat_template == tokenizer.chat_template
        assert loaded.get_vocab() == tokenizer.get_vocab()
        assert loaded.apply_chat_template(messages, tokenize=True, return_dict=False) == ids[0].tolist()
        assert loaded.apply_chat_template(messages[:1], tokenize=True, return_dict=False,
                                           add_generation_prompt=True) == prefix
        # save_pretrained 是推理资产往返，不是包含优化器/RNG 的完整训练恢复。
        model.eval()
        restored.eval()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            before = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            after = restored(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        torch.testing.assert_close(before, after, rtol=0, atol=0)


EXAMPLES = {"tensor": tensor_example, "autograd": autograd_example,
            "resume": resume_example, "lora": lora_example,
            "causal": causal_example, "transformers": transformers_example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=["all", *EXAMPLES], nargs="?", default="all")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    print(f"Python={platform.python_version()} torch={torch.__version__} device=cpu")
    selected = EXAMPLES if args.section == "all" else {args.section: EXAMPLES[args.section]}
    for name, run in selected.items():
        run()
        print(f"PASS {name}")
    print(f"PASS {len(selected)} sections; assertions only, no project training metrics")


if __name__ == "__main__":
    main()
