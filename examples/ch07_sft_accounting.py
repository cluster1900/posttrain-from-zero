"""Synthetic token-normalized SFT and complete-checkpoint lessons; CPU only."""
import copy
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F


def chapter_snippet():
    counts = [2, 6]
    nll_sums = [4.0, 6.0]
    token_mean = sum(nll_sums) / sum(counts)
    sample_mean = sum(n / c for n, c in zip(nll_sums, counts)) / len(counts)
    assert token_mean == 1.25
    assert sample_mean == 1.5


def accumulated_gradient():
    torch.manual_seed(7)
    logits = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])
    full = F.cross_entropy(logits, labels)
    reference, = torch.autograd.grad(full, logits)
    split = logits.detach().clone().requires_grad_()
    for section in (slice(0, 2), slice(2, 8)):
        (F.cross_entropy(split[section], labels[section], reduction="sum") / 8).backward()
    assert torch.allclose(split.grad, reference, atol=1e-12, rtol=1e-12)
    return reference, split.grad


def complete_checkpoint(directory):
    """A deliberately minimal selector, not the project's full hash validator."""
    candidates = []
    for path in Path(directory).glob("step_*"):
        suffix = path.name.removeprefix("step_")
        if path.is_dir() and suffix.isdigit() and (path / ".complete").is_file():
            candidates.append((int(suffix), path))
    return max(candidates, default=(None, None))[1]


def optimizer_resume():
    torch.manual_seed(3)
    parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)

    def update(p, opt):
        opt.zero_grad(set_to_none=True)
        (p - 2).square().sum().backward()
        opt.step()

    update(parameter, optimizer)
    weight, state = parameter.detach().clone(), copy.deepcopy(optimizer.state_dict())
    update(parameter, optimizer)
    restored = torch.nn.Parameter(weight.clone())
    restored_optimizer = torch.optim.AdamW([restored], lr=0.1)
    restored_optimizer.load_state_dict(state)
    update(restored, restored_optimizer)
    assert torch.equal(parameter, restored)
    weights_only = torch.nn.Parameter(weight.clone())
    fresh_optimizer = torch.optim.AdamW([weights_only], lr=0.1)
    update(weights_only, fresh_optimizer)
    assert not torch.equal(parameter, weights_only)
    return parameter.detach(), restored.detach(), weights_only.detach()


def demo():
    chapter_snippet()
    accumulated_gradient()
    optimizer_resume()
    with tempfile.TemporaryDirectory() as directory:
        old = Path(directory) / "step_10"
        old.mkdir()
        (old / ".complete").write_text("teaching marker")
        (Path(directory) / "step_20").mkdir()
        assert complete_checkpoint(directory) == old
    print("PASS: token-normalized gradients and optimizer resume; synthetic CPU data only")


if __name__ == "__main__":
    demo()
