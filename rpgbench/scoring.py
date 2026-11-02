"""Pure, deterministic stat, score and tier math (no torch)."""
import math

# Reference baselines: a value equal to its reference earns a stat of 100.
STR_REF_GFLOPS = 300.0
AGI_REF_MS = 0.5
INT_REF_TPS = 150.0
VIT_REF_GBPS = 40.0
END_REF_GFLOPS = 300.0

MIN_SCORE = 0.01

POTATO_MIN = 0.0
TOASTER_MIN = 60.0
WORKHORSE_MIN = 120.0
GAMING_RIG_MIN = 300.0
MINI_SUPERCOMPUTER_MIN = 600.0

TIERS = ("Potato", "Toaster", "Workhorse", "Gaming Rig", "Mini Supercomputer")
TIER_THRESHOLDS = (
    POTATO_MIN,
    TOASTER_MIN,
    WORKHORSE_MIN,
    GAMING_RIG_MIN,
    MINI_SUPERCOMPUTER_MIN,
)

STAT_NAMES = ("STR", "AGI", "INT", "VIT", "END")


def _positive_finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def _stat_from_ratio(ratio):
    stat = 100.0 * math.log2(1.0 + ratio)
    return stat if math.isfinite(stat) and stat > 0 else 0.0


def _stat(value, reference):
    if not _positive_finite(value):
        return 0.0
    return _stat_from_ratio(value / reference)


def compute_stats(str_gflops, agi_latency_ms, int_tps, vit_gbps, end_gflops):
    """Map raw measurements to the five RPG stats (higher is better)."""
    if _positive_finite(agi_latency_ms):
        agi = _stat_from_ratio(AGI_REF_MS / agi_latency_ms)
    else:
        agi = 0.0
    return {
        "STR": _stat(str_gflops, STR_REF_GFLOPS),
        "AGI": agi,
        "INT": _stat(int_tps, INT_REF_TPS),
        "VIT": _stat(vit_gbps, VIT_REF_GBPS),
        "END": _stat(end_gflops, END_REF_GFLOPS),
    }


def compute_score(stats):
    """Overall score: mean of the five stats, never below MIN_SCORE."""
    values = []
    for name in STAT_NAMES:
        value = stats.get(name, 0.0)
        if not _positive_finite(value):
            value = 0.0
        values.append(float(value))
    mean = sum(values) / len(values)
    if not math.isfinite(mean):
        mean = 1e300
    return max(MIN_SCORE, mean)


def score_to_tier(score):
    """Return the tier whose lower threshold is the highest one <= score."""
    if not isinstance(score, (int, float)) or not score == score:
        return TIERS[0]
    tier = TIERS[0]
    for name, threshold in zip(TIERS, TIER_THRESHOLDS):
        if score >= threshold:
            tier = name
    return tier
