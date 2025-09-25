"""Learning-rate schedule helpers."""

from __future__ import annotations


def lr_from_schedule(
    epoch: int, base: float, steps: list[int], values: list[float], schedule_type: str = "step"
) -> float:
    if schedule_type == "constant":
        return float(base)
    buckets = [(-1, values[0])]
    for idx, e in enumerate(steps):
        v = values[min(idx, len(values) - 1)]
        buckets.append((e, v))
    last_v = values[-1]
    current = values[0]
    for e, v in buckets[1:]:
        if epoch < e:
            break
        current = v
    if len(steps) > 0 and epoch >= steps[-1]:
        current = last_v
    return float(current)
