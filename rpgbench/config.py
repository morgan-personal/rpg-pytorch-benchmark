"""Tunable constants: time budgets, seeds, race target and allocation caps."""

# Seconds each sub-benchmark may spend, per mode. Every quick budget must stay
# strictly smaller than the matching full budget.
BUDGETS = {
    "full": {
        "STR": 36.0,
        "AGI": 24.0,
        "INT": 36.0,
        "VIT": 26.0,
        "END": 52.0,
        "RACE": 45.0,
    },
    "quick": {
        "STR": 3.0,
        "AGI": 2.5,
        "INT": 3.5,
        "VIT": 2.5,
        "END": 6.5,
        "RACE": 8.0,
    },
}

# Length of the sustained-load loop and of its first/last measurement windows.
END_DURATION_S = {"full": 50.0, "quick": 6.0}
END_WINDOW_S = {"full": 10.0, "quick": 1.2}

SEED = 2026
RACE_TARGET_ACCURACY = 0.90
MIN_REPEATS = 3
WARMUP_ITERS = 1

GIB = 1024 ** 3

# Allocation probe safety caps.
CPU_ALLOC_RAM_FRACTION = 0.25
CPU_ALLOC_MAX_BYTES = 4 * GIB
CPU_ALLOC_UNKNOWN_BYTES = 1 * GIB
GPU_ALLOC_FREE_FRACTION = 0.50
GPU_ALLOC_MAX_BYTES = 4 * GIB
GPU_ALLOC_UNKNOWN_BYTES = 2 * GIB
