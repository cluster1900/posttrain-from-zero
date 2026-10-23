"""Original argument/lifecycle simulation. Never runs a process or model command."""
from dataclasses import dataclass
import re
import shlex


def container_argv(container, command, timeout=120):
    if not isinstance(container, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}', container):
        raise ValueError('invalid trusted container name')
    if not isinstance(command, str) or '\x00' in command:
        raise ValueError('command must be a string without NUL')
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError('invalid teaching timeout')
    wrapped = f'timeout --signal=TERM --kill-after=10 {timeout} bash -c {shlex.quote(command)}'
    return ['docker', 'exec', container, 'bash', '-c', wrapped]


@dataclass
class Probe:
    command_exit: int | None
    reward: float | None
    infra_error: str | None = None
    timed_out: bool = False
    oom_killed: bool = False


def result_kind(probe):
    if probe.infra_error is not None:
        return 'infrastructure failure'
    if probe.timed_out or probe.oom_killed or probe.command_exit is None or probe.reward not in (0.0, 1.0):
        return 'verifier incomplete'
    return 'task passed' if probe.reward == 1.0 else 'task failed'


def cleanup_verified(delete_requested, readbacks):
    """Every expected resource is explicitly absent; unknown is never absent."""
    expected = {'job', 'sandbox_vm', 'disk', 'firewall'}
    return delete_requested is True and set(readbacks) == expected and all(
        value == 'ABSENT' for value in readbacks.values())


def chapter_snippet():
    command = 'echo "a b"; exit 7'
    argv = container_argv('teaching-container', command)
    assert shlex.split(argv[-1])[-1] == command
    assert result_kind(Probe(0, 0.0)) == 'task failed'
    assert result_kind(Probe(0, None, 'transport timeout')) == 'infrastructure failure'
    return argv[:5]


if __name__ == '__main__':
    print(chapter_snippet())
