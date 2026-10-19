"""Exact standalone Python blocks shown in chapter 1. No network or GPU needed."""

def snippet_1():
    import torch
    ids = torch.tensor([[1, 3, 2]], dtype=torch.long)
    embedding = torch.nn.Embedding(8, 4)
    hidden = embedding(ids)
    assert ids.shape == (1, 3)
    assert hidden.shape == (1, 3, 4)
    assert ids.device.type == "cpu"

def snippet_2():
    import torch
    x = torch.zeros(2, 3, dtype=torch.float32)
    y = x.to(torch.bfloat16)
    assert x.numel() * x.element_size() == 24
    assert y.numel() * y.element_size() == 12

def snippet_3():
    import torch
    w = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([w], lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    loss = (w - 2.0).square()
    loss.backward()
    assert w.item() == 0.0                 # 仍未更新
    assert w.grad.item() == -4.0
    optimizer.step()
    assert torch.isclose(w.detach(), torch.tensor(0.4))

def snippet_4():
    import torch
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 2)
        def forward(self, x):
            return self.proj(x)

    model = TinyModel()
    assert "proj.weight" in model.state_dict()
    assert model(torch.ones(1, 3)).shape == (1, 2)

def snippet_5():
    from torch.utils.checkpoint import checkpoint
    # block 是已定义的模块，x 是模块输入
    # non-reentrant 模式允许输入不要求梯度、内部参数需要梯度
    # y = checkpoint(block, x, use_reentrant=False)

def snippet_6():
    import torch
    import torch.nn.functional as F
    logits = torch.zeros(1, 4, 6, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3]])
    targets = labels[:, 1:]
    count = (targets != -100).sum()
    assert count.item() == 2
    loss_sum = F.cross_entropy(
        logits[:, :-1].reshape(-1, 6), targets.reshape(-1),
        ignore_index=-100, reduction="sum",
    )
    loss = loss_sum / count
    loss.backward()

if __name__ == "__main__":
    snippet_1()
    snippet_2()
    snippet_3()
    snippet_4()
    snippet_5()
    snippet_6()
