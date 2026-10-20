"""Offline teaching constructions, not project rollouts or a complete RL trainer."""
import torch


def chapter_snippet():
    import torch
    rewards = torch.tensor([[0., 0., 0., 0.], [0., 1., 0., 1.]])
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    useful = rewards.max(dim=1).values != rewards.min(dim=1).values
    assert useful.tolist() == [False, True]
    assert advantages[1].tolist() == [-0.5, 0.5, -0.5, 0.5]


def group_signal(scores, infra, truncated):
    """Minimal illustration of documented filtering order; no sampling or commands."""
    if not (len(scores) == len(infra) == len(truncated)):
        raise ValueError('fields must have equal length')
    valid = [(i, float(s)) for i, s in enumerate(scores) if not infra[i]]
    if any(s not in (0., 1.) for _, s in valid):
        raise ValueError('teaching contract requires binary rewards')
    if len(valid) < 2 or len({s for _, s in valid}) == 1:
        return None
    mean = sum(s for _, s in valid) / len(valid)
    return [(i, s - mean, not truncated[i]) for i, s in valid]


def demo():
    chapter_snippet()
    signal = group_signal([0, 1, 0], [True, False, False], [False, False, True])
    assert signal == [(1, .5, True), (2, -.5, False)]
    assert group_signal([0, 1], [True, False], [False, False]) is None
    assert group_signal([1, 1], [False, False], [False, False]) is None
    print('PASS: centered groups, infrastructure exclusion, truncated baseline; teaching only')


if __name__ == '__main__':
    demo()
