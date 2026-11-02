import math
import statistics

import pytest

torch = pytest.importorskip("torch")

from rpgbench import config, devices, workloads  # noqa: E402

GIB = 1024 ** 3


@pytest.fixture(scope="module")
def ctx():
    return devices.cpu_context(torch, 8 * GIB)


def _common(res, budget):
    assert res["budget_s"] == budget
    assert res["elapsed_s"] >= 0
    assert res["threads"] >= 1


def _median_ok(res, key, samples_key="samples"):
    samples = res[samples_key]
    assert len(samples) >= 3
    assert res["warmup_iters"] >= 1
    assert math.isclose(res[key], statistics.median(samples), rel_tol=1e-9)


def test_str(ctx):
    res = workloads.run_str(torch, ctx, 1.5, "quick", None)
    _common(res, 1.5)
    _median_ok(res, "gflops_fp32")
    assert res["gflops_fp16"] is None
    assert res["fp16_note"]
    assert res["gflops_fp32"] > 0
    assert res["elapsed_s"] <= 2 * 1.5 + 3


def test_agi(ctx):
    res = workloads.run_agi(torch, ctx, 1.5, "quick", None)
    _common(res, 1.5)
    _median_ok(res, "latency_ms")
    assert 1 <= res["batch_size"] <= 8
    assert res["elapsed_s"] <= 2 * 1.5 + 3


def test_int(ctx):
    res = workloads.run_int(torch, ctx, 2.0, "quick", None)
    _common(res, 2.0)
    _median_ok(res, "tokens_per_s")
    assert 16 <= res["gen_tokens"] <= 128
    assert set(res["model"]) == {"vocab", "d_model", "layers", "heads"}
    assert res["elapsed_s"] <= 2 * 2.0 + 3


def test_vit(ctx):
    res = workloads.run_vit(torch, ctx, 1.5, "quick", None)
    _common(res, 1.5)
    _median_ok(res, "copy_gbps")
    assert res["largest_alloc_gb"] <= res["alloc_cap_gb"]
    assert res["alloc_cap_gb"] == pytest.approx(2.0)
    assert res["convention"]


def test_vit_small_cap_respected():
    small = devices.cpu_context(torch, 64 * 1024 * 1024)
    small.alloc_cap_bytes = 200 * 1024 * 1024
    res = workloads.run_vit(torch, small, 1.5, "quick", None)
    assert res["largest_alloc_gb"] <= res["alloc_cap_gb"]
    assert res["largest_alloc_gb"] > 0


def test_end_scaled(ctx):
    res = workloads.run_end(torch, ctx, 1.5, "quick", None)
    _common(res, 1.5)
    for key in ("first_window_gflops", "last_window_gflops",
                "sustained_gflops", "throttle_pct", "duration_s",
                "window_s"):
        assert key in res and math.isfinite(res[key])
    assert res["duration_s"] < config.END_DURATION_S["full"]
    assert res["window_s"] <= config.END_WINDOW_S["quick"]
    assert res["sustained_gflops"] > 0


def test_matmul_inputs_are_float32_on_cpu(ctx):
    """The shared matmul input helper yields float32 tensors on the device."""
    a, b = workloads._matrices(torch, ctx, 8)
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    assert a.device.type == "cpu"
    assert b.device.type == "cpu"
    assert a.device == ctx.device
    assert b.device == ctx.device
