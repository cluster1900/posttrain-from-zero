"""Original CPU DPPO exercise; tiny supplied probabilities, not model training."""
import math
import torch


def centered_items(rewards, infra, truncated):
    if not (len(rewards) == len(infra) == len(truncated)):
        raise ValueError('parallel lists must have equal lengths')
    if any(type(x) is not bool for x in infra + truncated):
        raise ValueError('flags must be booleans')
    valid = [i for i, failed in enumerate(infra) if not failed]
    if any(rewards[i] not in (0, 1) for i in valid):
        raise ValueError('valid outcomes must be binary')
    if len(valid) < 2 or len({rewards[i] for i in valid}) == 1:
        return []
    baseline = sum(rewards[i] for i in valid) / len(valid)
    return [(i, rewards[i] - baseline) for i in valid if not truncated[i]]


def dppo(new_logp, old_logp, advantage, response_mask, threshold=0.1):
    """Return token-normalized loss and a detached directional binary-TV mask."""
    if not (new_logp.ndim == 1 and new_logp.shape == old_logp.shape == advantage.shape == response_mask.shape):
        raise ValueError('equal one-dimensional shapes required')
    if response_mask.dtype != torch.bool:
        raise ValueError('response mask must be boolean')
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError('threshold must be finite and nonnegative')
    if not all(torch.isfinite(t).all() for t in (new_logp, old_logp, advantage)):
        raise ValueError('non-finite inputs')
    with torch.no_grad():
        divergence = (new_logp.clamp(-30, 0).exp() - old_logp.clamp(-30, 0).exp()).abs()
        ratio = (new_logp - old_logp).clamp(-20, 20).exp()
        harmful = ((advantage > 0) & (ratio > 1)) | ((advantage < 0) & (ratio < 1))
        keep = response_mask & ~(harmful & (divergence > threshold))
    ratio = (new_logp - old_logp).clamp(-20, 20).exp()
    denominator = response_mask.sum().clamp_min(1)
    return (-advantage * ratio * keep).sum() / denominator, keep


def bounded_requests(useful_per_round, prompts=4, max_refills=3):
    """Pure accounting: no sampling/network; supplied round counts are fictional."""
    if type(prompts) is not int or prompts <= 0 or type(max_refills) is not int or max_refills < 0:
        raise ValueError('invalid sampling bounds')
    requests, useful = [], 0
    for count in useful_per_round[:max_refills + 1]:
        need = prompts - useful
        if need <= 0:
            break
        if type(count) is not int or not 0 <= count <= need:
            raise ValueError('useful groups cannot exceed requested groups')
        requests.append(need)
        useful += count
    return requests


def chapter_snippet():
    rewards = [1, 0, 0, 0]
    infra = [False, False, False, True]
    truncated = [False, False, True, False]
    items = centered_items(rewards, infra, truncated)
    assert [i for i, _ in items] == [0, 1]
    assert abs(items[0][1] - 2/3) < 1e-12
    assert abs(items[1][1] + 1/3) < 1e-12
    return items


if __name__ == '__main__':
    print(chapter_snippet())
    old = torch.tensor([0.2, 0.2, 0.2, 0.2], dtype=torch.float64).log()
    new = torch.tensor([0.4, 0.1, 0.4, 0.1], dtype=torch.float64).log().requires_grad_()
    loss, keep = dppo(new, old, torch.tensor([1., 1., -1., -1.]), torch.ones(4, dtype=torch.bool))
    loss.backward()
    print('keep:', keep.tolist(), 'loss:', loss.item(), 'gradient:', new.grad.tolist())
