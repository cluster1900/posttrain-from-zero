"""Original offline quota exercise. No cloud SDK, requests or resource creation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Request:
    worker_cpus: int = 12
    worker_disk_gb: int = 100
    sandbox_cpus: int = 0
    sandbox_disk_gb: int = 0
    gpus: int = 1
    strategy: str = 'SPOT'


def deficits(request, quotas):
    """Teaching policy: every required quota must be present and finite."""
    numbers = (request.worker_cpus, request.worker_disk_gb, request.sandbox_cpus,
               request.sandbox_disk_gb, request.gpus)
    if any(type(n) is not int or n < 0 for n in numbers):
        raise ValueError('resource counts must be nonnegative integers')
    if request.strategy not in ('SPOT', 'STANDARD'):
        raise ValueError('unknown scheduling strategy')
    required = {'compute/global_cpu': request.worker_cpus + request.sandbox_cpus,
                'compute/ssd_gb': request.worker_disk_gb + request.sandbox_disk_gb}
    if request.gpus:
        required['vertex/' + request.strategy.lower() + '_a100'] = request.gpus
    missing = []
    for key, demand in required.items():
        value = quotas.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            missing.append(key + ': unknown')
            continue
        limit, usage = value
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in value):
            missing.append(key + ': invalid')
        elif usage + demand > limit:
            missing.append(key + ': insufficient')
    return missing


def eligible_to_request(request, first, second):
    """Two matching snapshots are not a reservation or a submission."""
    return first == second and not deficits(request, first) and not deficits(request, second)


def chapter_snippet():
    limits = {'compute/global_cpu': (12, 0), 'compute/ssd_gb': (250, 0),
              'vertex/spot_a100': (8, 0), 'compute/a100': (0, 0)}
    assert eligible_to_request(Request(), limits, dict(limits))
    assert deficits(Request(worker_disk_gb=300), limits) == ['compute/ssd_gb: insufficient']
    return 'eligible is not submitted'


if __name__ == '__main__':
    print(chapter_snippet())
