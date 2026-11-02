import statistics

from rpgbench import timing


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_measure_minimums_when_budget_tiny():
    clock = FakeClock()
    calls = []

    def batch():
        calls.append(1)
        clock.advance(10.0)

    m = timing.measure(batch, budget_s=1.0, start_time=clock(), clock=clock)
    assert m.warmup_iters >= 1
    assert len(m.samples_s) == 3
    assert len(calls) == m.warmup_iters + 3


def test_measure_fills_budget_without_overrunning():
    clock = FakeClock()

    def batch():
        clock.advance(1.0)

    m = timing.measure(batch, budget_s=20.0, start_time=clock(), clock=clock)
    assert 3 <= len(m.samples_s) <= 200
    assert m.elapsed_s <= 20.0
    assert m.elapsed_s >= 18.0


def test_measure_counts_time_before_call():
    clock = FakeClock()
    start = clock()
    clock.advance(9.0)

    def batch():
        clock.advance(1.0)

    m = timing.measure(batch, budget_s=10.0, start_time=start, clock=clock)
    assert len(m.samples_s) == 3


def test_measure_caps_repeats():
    clock = FakeClock()

    def batch():
        clock.advance(0.001)

    m = timing.measure(batch, budget_s=1000.0, start_time=clock(),
                       clock=clock, max_repeats=7)
    assert len(m.samples_s) == 7


def test_median_matches_statistics():
    values = [3.0, 1.0, 2.0, 10.0]
    assert timing.median(values) == statistics.median(values)


def test_calibrate_inner_doubles_until_target():
    clock = FakeClock()

    def once():
        clock.advance(0.01)

    inner = timing.calibrate_inner(once, lambda: None, 0.1, clock=clock)
    assert inner == 16


def test_calibrate_inner_respects_max():
    clock = FakeClock()
    inner = timing.calibrate_inner(lambda: clock.advance(1e-9),
                                   lambda: None, 100.0, clock=clock,
                                   max_inner=8)
    assert inner == 8
