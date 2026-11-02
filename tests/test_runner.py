import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import make_fake_torch
from rpgbench import devices, results, runner
from rpgbench.precheck import PrecheckResult

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_result.json"
ORDER = ("STR", "AGI", "INT", "VIT", "END", "RACE")


@pytest.fixture(scope="module")
def sample():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def make_steps(sample, fail=None, fail_label="AMD ROCm", interrupt=None):
    """Fake steps replaying fixture metrics; optionally failing on a GPU."""
    raw = sample["results"]["cpu"]["raw"]

    def build(name):
        def step(torch_mod, ctx, budget_s, mode, out):
            if ctx.label == fail_label and name == fail:
                raise RuntimeError("op not supported (C:\\Users\\x\\k.py)")
            if ctx.label == fail_label and name == interrupt:
                raise KeyboardInterrupt
            res = copy.deepcopy(raw[name])
            res["budget_s"] = budget_s
            return res

        return step

    return lambda: [(name, build(name)) for name in ORDER]


def cpu_ctx_fn(torch_mod, ram):
    return SimpleNamespace(label="CPU", gpu_name=None)


def gpu_ctx_fn(torch_mod, info):
    return SimpleNamespace(label=info.label, gpu_name="Fake GPU")


def hw_fn_for(sample):
    def hw(torch_mod, label, gpu_name, import_fn=None):
        hardware = dict(sample["hardware"])
        hardware["backend"] = label
        hardware["gpu_name"] = gpu_name
        return hardware

    return hw


def detect_gpu(torch_mod):
    return devices.BackendInfo("AMD ROCm", "cuda")


def run(sample, mode="quick", cpu_only=False, detect_fn=None, context_fn=None,
        steps_fn=None):
    lines = []
    cfg = runner.RunConfig(mode=mode, cpu_only=cpu_only, nickname="tester",
                           file_part="tester")
    pre = PrecheckResult(warnings=["Running on battery: plug in"])
    doc = runner.run_all(
        cfg, torch_mod=make_fake_torch(), precheck=pre,
        detect_fn=detect_fn or (lambda t: devices.BackendInfo("CPU", "cpu")),
        context_fn=context_fn or gpu_ctx_fn, cpu_context_fn=cpu_ctx_fn,
        steps_fn=steps_fn or make_steps(sample), hw_fn=hw_fn_for(sample),
        ram_fn=lambda: None, out=lambda t="", end="\n": lines.append(t))
    return doc, lines


def test_cpu_only_machine_gpu_is_null(sample):
    doc, _ = run(sample)
    assert doc["results"]["gpu"] is None
    assert doc["primary"] == "cpu"
    assert doc["hardware"]["backend"] == "CPU"
    assert results.validate_document(doc) == []
    cpu = doc["results"]["cpu"]
    assert cpu["device_label"] == "CPU"
    assert cpu["training_race"] == cpu["raw"]["RACE"]
    assert cpu["raw"]["RACE"]["seed"] == 2026
    assert doc["warnings"] == ["Running on battery: plug in"]


def test_gpu_detected_gives_both_results(sample):
    doc, _ = run(sample, detect_fn=detect_gpu)
    cpu, gpu = doc["results"]["cpu"], doc["results"]["gpu"]
    assert gpu["device_label"] == "AMD ROCm"
    assert cpu["device_label"] == "CPU"
    for res in (cpu, gpu):
        assert res["stats"] and res["score"] > 0 and res["tier"]
        assert res["comparison"] and res["training_race"]
    assert doc["primary"] == "gpu"
    assert doc["hardware"]["backend"] == "AMD ROCm"
    assert doc["hardware"]["gpu_name"] == "Fake GPU"
    assert results.validate_document(doc) == []


def test_cpu_only_skips_detection(sample):
    def boom(torch_mod):
        raise AssertionError("detection must be skipped")

    doc, _ = run(sample, cpu_only=True, detect_fn=boom)
    assert doc["results"]["gpu"] is None
    assert doc["hardware"]["backend"] == "CPU"


def test_gpu_init_failure_falls_back(sample):
    def broken(torch_mod, info):
        raise RuntimeError("driver not responding")

    doc, _ = run(sample, detect_fn=detect_gpu, context_fn=broken)
    gpu = doc["results"]["gpu"]
    assert gpu["score"] is None
    assert gpu["error"]["step"] == "GPU initialisation"
    assert any("GPU initialisation" in w and "AMD ROCm" in w
               for w in doc["warnings"])
    assert doc["primary"] == "cpu"
    assert doc["results"]["cpu"]["score"] > 0
    assert results.validate_document(doc) == []


@pytest.mark.parametrize("step", ["STR", "INT", "END", "RACE"])
def test_gpu_step_failure_falls_back(sample, step):
    doc, _ = run(sample, detect_fn=detect_gpu,
                 steps_fn=make_steps(sample, fail=step))
    gpu = doc["results"]["gpu"]
    assert gpu["score"] is None and gpu["stats"] is None
    assert gpu["error"]["step"] == step
    assert any("'%s'" % step in w and "AMD ROCm" in w for w in doc["warnings"])
    assert doc["primary"] == "cpu"
    assert doc["results"]["cpu"]["score"] > 0
    assert step not in gpu["raw"]
    done = ORDER[:ORDER.index(step)]
    assert all(name in gpu["raw"] for name in done)
    assert results.validate_document(doc) == []


def test_failure_messages_are_scrubbed(sample):
    doc, _ = run(sample, detect_fn=detect_gpu,
                 steps_fn=make_steps(sample, fail="INT"))
    text = json.dumps(doc)
    assert "C:\\\\Users" not in text and "k.py" not in text


def test_cpu_failure_raises_benchmark_error(sample):
    def steps():
        def bad(torch_mod, ctx, budget_s, mode, out):
            raise ValueError("nope")

        return [("STR", bad)]

    with pytest.raises(runner.BenchmarkError):
        run(sample, steps_fn=steps)


def test_keyboard_interrupt_propagates(sample):
    with pytest.raises(KeyboardInterrupt):
        run(sample, detect_fn=detect_gpu,
            steps_fn=make_steps(sample, interrupt="AGI"))


def test_detection_raising_is_contained(sample):
    def boom(torch_mod):
        raise RuntimeError("x")

    doc, _ = run(sample, detect_fn=boom)
    assert doc["results"]["gpu"] is None


def test_reseeds_before_each_step(sample):
    seeds = []
    fake = make_fake_torch()
    fake.manual_seed = seeds.append
    ctx = SimpleNamespace(label="CPU")
    runner.run_suite(fake, ctx, "quick", make_steps(sample)(),
                     lambda t="", end="\n": None)
    assert len(seeds) == len(ORDER)
    assert set(seeds) == {2026}


def test_suite_budgets_come_from_config(sample):
    seen = {}

    def step(torch_mod, ctx, budget_s, mode, out):
        seen[mode] = budget_s
        return {"budget_s": budget_s, "elapsed_s": 0.0}

    runner.run_suite(make_fake_torch(), SimpleNamespace(label="CPU"), "quick",
                     [("STR", step)], lambda t="", end="\n": None)
    runner.run_suite(make_fake_torch(), SimpleNamespace(label="CPU"), "full",
                     [("STR", step)], lambda t="", end="\n": None)
    assert seen["quick"] < seen["full"]


def test_metrics_from_raw_handles_missing():
    assert runner.metrics_from_raw({}) == (None,) * 5
