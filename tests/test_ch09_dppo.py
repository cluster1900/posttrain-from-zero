"""Offline boundary regressions for the three original chapter exercises."""
import math
import shlex
import pytest
import torch
from examples.ch08_preflight import Request, deficits, eligible_to_request
from examples.ch09_dppo import centered_items, dppo, bounded_requests
from examples.ch10_sandbox_contract import container_argv, Probe, result_kind, cleanup_verified


def test_baseline_excludes_infra_but_includes_truncation():
    out = centered_items([1, 0, 0, 999], [False, False, False, True], [False, False, True, False])
    assert out == [(0, 1-1/3), (1, -1/3)]
    assert centered_items([1, 1], [False, False], [False, False]) == []
    assert centered_items([1, 0], [False, True], [False, False]) == []
    assert centered_items([1, 0], [False, False], [True, True]) == []


def test_dppo_signs_denominator_and_gradient():
    # Binary fractions make the TV boundary exact: 0.25 -> 0.375, tau=0.125.
    old = torch.tensor([.25]*6, dtype=torch.float64).log()
    new = torch.tensor([.5, .125, .5, .125, .375, .5], dtype=torch.float64).log().requires_grad_()
    advantage = torch.tensor([1, 1, -1, -1, 1, 1], dtype=torch.float64)
    mask = torch.tensor([True]*5+[False])
    loss, keep = dppo(new, old, advantage, mask, threshold=.125)
    assert keep.tolist() == [False, True, True, True, True, False]
    loss.backward()
    expected = -advantage * (new.detach()-old).exp() * keep / 5
    assert torch.allclose(new.grad, expected, atol=1e-12, rtol=0)
    assert not keep.requires_grad
    assert loss.item() == pytest.approx((-0.5+2+0.5-1.5)/5)


def test_dppo_scalar_oracle_grid():
    # Python scalar independent decision rule for a grid away from boundary roundoff.
    old_p, new_p, advantages = [], [], []
    for p in (.125, .25, .5):
        for q in (.0625, .25, .75):
            for a in (-.75, 0., .25):
                old_p.append(p); new_p.append(q); advantages.append(a)
    old = torch.tensor(old_p, dtype=torch.float64).log()
    new = torch.tensor(new_p, dtype=torch.float64).log().requires_grad_()
    adv = torch.tensor(advantages, dtype=torch.float64)
    mask = torch.ones(len(adv), dtype=torch.bool)
    loss, keep = dppo(new, old, adv, mask)
    expected_keep = [not(abs(p-q)>.1 and ((a>0 and q>p) or (a<0 and q<p)))
                     for p,q,a in zip(old_p,new_p,advantages)]
    expected_loss = sum(-a*q/p*k for p,q,a,k in zip(old_p,new_p,advantages,expected_keep))/len(adv)
    assert keep.tolist() == expected_keep
    assert loss.item() == pytest.approx(expected_loss)
    loss.backward()
    expected_grad = torch.tensor([-a*q/p*k/len(adv) for p,q,a,k in zip(old_p,new_p,advantages,expected_keep)])
    assert torch.allclose(new.grad, expected_grad.double(), atol=1e-8)


def test_empty_loss_and_nonfinite():
    new = torch.empty(0, dtype=torch.float64, requires_grad=True)
    loss, keep = dppo(new, new.detach(), new.detach(), torch.empty(0, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and len(keep) == 0
    with pytest.raises(ValueError):
        dppo(torch.tensor([float('nan')]), torch.zeros(1), torch.ones(1), torch.ones(1,dtype=torch.bool))


def test_bounded_refills_are_not_guaranteed_full_batch():
    assert bounded_requests([1, 1, 0, 0, 2]) == [4, 3, 2, 2]
    assert bounded_requests([4, 0]) == [4]
    assert bounded_requests([0]*9, max_refills=0) == [4]
    with pytest.raises(ValueError):
        bounded_requests([5])

