"""Budget-driven measurement helpers (torch-free; clock and sync injected)."""
import statistics
import time
from dataclasses import dataclass

from rpgbench import config


@dataclass
class Measurement:
    samples_s: list
    warmup_iters: int
    elapsed_s: float


def median(values):
    """Median used for every reported value (statistics.median)."""
    return statistics.median(values)


def calibrate_inner(run_once, sync, target_s, clock=time.perf_counter,
                    max_inner=1 << 20):
    """Double the inner-iteration count until one batch takes >= target_s."""
    inner = 1
    while True:
        begin = clock()
        for _ in range(inner):
            run_once()
        sync()
        took = clock() - begin
        if took >= target_s or inner >= max_inner:
            return inner
        inner = min(inner * 2, max_inner)


def measure(run_batch, *, budget_s, start_time, clock=time.perf_counter,
            warmup=config.WARMUP_ITERS, min_repeats=config.MIN_REPEATS,
            max_repeats=200):
    """Warm up, then time repeated batches while the next one fits the budget.

    run_batch must itself synchronise the device so each sample covers
    completed work. start_time is the caller's entry time, so calibration
    done before this call counts against the budget.
    """
    warmup = max(1, int(warmup))
    for _ in range(warmup):
        run_batch()
    samples = []
    while len(samples) < max_repeats:
        begin = clock()
        run_batch()
        took = clock() - begin
        samples.append(took)
        if len(samples) >= min_repeats:
            if (clock() - start_time) + took > budget_s:
                break
    return Measurement(samples, warmup, clock() - start_time)
