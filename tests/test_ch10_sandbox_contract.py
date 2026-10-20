"""Offline boundary regressions for the three original chapter exercises."""
import math
import shlex
import pytest
import torch
from examples.ch08_preflight import Request, deficits, eligible_to_request
from examples.ch09_dppo import centered_items, dppo, bounded_requests
from examples.ch10_sandbox_contract import container_argv, Probe, result_kind, cleanup_verified


@pytest.mark.parametrize('command', ['echo "a b"; exit 7', '$(touch /tmp/never-run)',
                                    "x'; echo break; #", 'line1\nline2', '--privileged'])
def test_untrusted_command_remains_single_container_argument(command):
    argv = container_argv('demo', command)
    assert argv[:5] == ['docker', 'exec', 'demo', 'bash', '-c']
    inner = shlex.split(argv[-1])
    assert inner == ['timeout', '--signal=TERM', '--kill-after=10', '120', 'bash', '-c', command]


@pytest.mark.parametrize('container,timeout', [('--privileged',120), ('x; bad',120), ('ok',0), ('ok',True)])
def test_transport_rejects_invalid_trusted_fields(container,timeout):
    with pytest.raises(ValueError):
        container_argv(container, 'true', timeout)


def test_probe_zero_reward_is_not_success_and_cleanup_requires_readback():
    assert result_kind(Probe(0,0)) == 'task failed'
    assert result_kind(Probe(0,1)) == 'task passed'
    assert result_kind(Probe(7,1)) == 'task passed'
    assert result_kind(Probe(0,1,timed_out=True)) == 'verifier incomplete'
    assert result_kind(Probe(137,1,oom_killed=True)) == 'verifier incomplete'
    assert result_kind(Probe(0,None)) == 'verifier incomplete'
    assert result_kind(Probe(0,1,'timeout')) == 'infrastructure failure'
    readbacks = dict.fromkeys(['job','sandbox_vm','disk','firewall'], 'ABSENT')
    assert cleanup_verified(True,readbacks)
    assert not cleanup_verified(False,readbacks)
    assert not cleanup_verified(True,{**readbacks,'disk':'UNKNOWN'})
    assert not cleanup_verified(True,{'job':'ABSENT'})
