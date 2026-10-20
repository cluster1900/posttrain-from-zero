"""Offline boundary regressions for the three original chapter exercises."""
import math
import shlex
import pytest
import torch
from examples.ch08_preflight import Request, deficits, eligible_to_request
from examples.ch09_dppo import centered_items, dppo, bounded_requests
from examples.ch10_sandbox_contract import container_argv, Probe, result_kind, cleanup_verified


def quotas():
    return {'compute/global_cpu': (12, 0), 'compute/ssd_gb': (250, 0),
            'vertex/spot_a100': (8, 0), 'vertex/standard_a100': (0, 0), 'compute/a100': (0, 0)}


def test_namespace_and_double_snapshot():
    q = quotas()
    assert eligible_to_request(Request(), q, dict(q))
    assert not eligible_to_request(Request(), q, {**q, 'compute/global_cpu': (12, 1)})
    assert deficits(Request(strategy='STANDARD'), q) == ['vertex/standard_a100: insufficient']
    assert deficits(Request(sandbox_cpus=4, sandbox_disk_gb=200), q) == [
        'compute/global_cpu: insufficient', 'compute/ssd_gb: insufficient']


@pytest.mark.parametrize('value', [None, (), (12,), (-1, 0), (12, float('nan')), ('12', 0), (True, 0)])
def test_unknown_quota_fails_closed(value):
    q = quotas()
    q['compute/global_cpu'] = value
    assert deficits(Request(), q)


@pytest.mark.parametrize('spec', [Request(worker_cpus=-1), Request(gpus=True), Request(strategy='other')])
def test_bad_request(spec):
    with pytest.raises(ValueError):
        deficits(spec, quotas())

