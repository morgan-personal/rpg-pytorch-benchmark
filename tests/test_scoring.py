import math

import pytest

from rpgbench import scoring as sc

BASE = dict(str_gflops=50.0, agi_latency_ms=8.0, int_tps=40.0,
            vit_gbps=12.0, end_gflops=45.0)


def _score(**overrides):
    args = dict(BASE)
    args.update(overrides)
    return sc.compute_score(sc.compute_stats(**args))


def test_deterministic():
    assert sc.compute_stats(**BASE) == sc.compute_stats(**BASE)
    assert _score() == _score()


def test_stats_keys_finite_non_negative():
    stats = sc.compute_stats(**BASE)
    assert list(stats) == ["STR", "AGI", "INT", "VIT", "END"]
    assert all(math.isfinite(v) and v >= 0 for v in stats.values())


@pytest.mark.parametrize("key", ["str_gflops", "int_tps", "vit_gbps",
                                 "end_gflops"])
def test_monotonic_increasing_inputs(key):
    last = -1.0
    for factor in (0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 1000.0):
        value = _score(**{key: BASE[key] * factor})
        assert value >= last
        last = value


def test_monotonic_agi_latency_falling():
    last = -1.0
    for latency in (1000.0, 100.0, 10.0, 1.0, 0.1, 0.001):
        value = _score(agi_latency_ms=latency)
        assert value >= last
        last = value


@pytest.mark.parametrize("bad", [None, 0, -1.0, float("nan"), float("inf"),
                                 "x", True])
def test_bad_inputs_give_zero_stat(bad):
    stats = sc.compute_stats(bad, bad, bad, bad, bad)
    assert all(v == 0.0 for v in stats.values())
    assert sc.compute_score(stats) == sc.MIN_SCORE


def test_score_floor_and_finite():
    assert sc.compute_score({}) == sc.MIN_SCORE
    huge = sc.compute_stats(1e300, 1e-300, 1e300, 1e300, 1e300)
    assert math.isfinite(sc.compute_score(huge))


def test_tier_order():
    assert sc.TIERS == ("Potato", "Toaster", "Workhorse", "Gaming Rig",
                        "Mini Supercomputer")
    thresholds = [sc.POTATO_MIN, sc.TOASTER_MIN, sc.WORKHORSE_MIN,
                  sc.GAMING_RIG_MIN, sc.MINI_SUPERCOMPUTER_MIN]
    assert thresholds == sorted(thresholds)
    assert len(set(thresholds)) == 5


def test_every_tier_reachable_and_threshold_equality():
    for name, threshold in zip(sc.TIERS, sc.TIER_THRESHOLDS):
        assert sc.score_to_tier(threshold) == name
        assert sc.score_to_tier(threshold + 0.001) == name
    assert sc.score_to_tier(sc.TOASTER_MIN - 1e-9) == "Potato"
    assert sc.score_to_tier(sc.GAMING_RIG_MIN - 1e-9) == "Workhorse"


def test_extremes():
    assert sc.score_to_tier(1e-9) == "Potato"
    assert sc.score_to_tier(sc.MIN_SCORE) == "Potato"
    assert sc.score_to_tier(1e9 * sc.MINI_SUPERCOMPUTER_MIN) == (
        "Mini Supercomputer")
    assert sc.score_to_tier(float("nan")) == "Potato"


def test_all_tiers_reachable_from_raw_inputs():
    seen = set()
    for scale in [10 ** (i / 4.0) for i in range(-16, 25)]:
        stats = sc.compute_stats(
            BASE["str_gflops"] * scale, BASE["agi_latency_ms"] / scale,
            BASE["int_tps"] * scale, BASE["vit_gbps"] * scale,
            BASE["end_gflops"] * scale)
        seen.add(sc.score_to_tier(sc.compute_score(stats)))
    assert seen == set(sc.TIERS)
