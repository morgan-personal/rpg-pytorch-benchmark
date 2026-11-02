"""Orchestration: CPU suite, optional GPU suite, scoring, document.

Torch and every other heavy dependency are injected, so this module imports
nothing heavy.
"""
import random
from dataclasses import dataclass, field
from datetime import datetime

from rpgbench import config, devices, display, history, hwinfo, results
from rpgbench import scoring

STEP_ORDER = ("STR", "AGI", "INT", "VIT", "END", "RACE")


class BenchmarkError(Exception):
    """Raised when the mandatory CPU suite cannot be completed."""


@dataclass
class RunConfig:
    mode: str = "full"
    cpu_only: bool = False
    nickname: str = "player"
    file_part: str = "player"


@dataclass
class SuiteOutcome:
    raw: dict = field(default_factory=dict)
    training_race: object = None
    failed_step: object = None
    error_message: object = None


def default_steps():
    """Ordered (name, callable) steps; imports torch-using modules lazily."""
    from rpgbench import race, workloads

    steps = list(workloads.STEPS)
    steps.append(("RACE", race.run_race))
    return steps


def _seed(torch_mod):
    random.seed(config.SEED)
    seed_fn = getattr(torch_mod, "manual_seed", None)
    if callable(seed_fn):
        seed_fn(config.SEED)


def run_suite(torch_mod, ctx, mode, steps, out):
    """Run steps in order; stop at the first failing step."""
    outcome = SuiteOutcome()
    for name, fn in steps:
        budget = config.BUDGETS[mode][name]
        out("  [%s] %s ..." % (ctx.label, name))
        try:
            _seed(torch_mod)
            res = fn(torch_mod, ctx, budget, mode, out)
        except Exception as exc:
            outcome.failed_step = name
            outcome.error_message = "%s: %s" % (type(exc).__name__, exc)
            out("  [%s] %s FAILED - %s" % (ctx.label, name,
                                           outcome.error_message))
            break
        outcome.raw[name] = res
        if name == "RACE":
            outcome.training_race = res
        out(display.step_line(ctx.label, name, display.step_summary(name, res),
                              res.get("elapsed_s", 0.0)))
    return outcome


def metrics_from_raw(raw):
    """Pull the five scoring inputs out of the raw metrics."""
    def pick(step, key):
        entry = raw.get(step)
        return entry.get(key) if isinstance(entry, dict) else None

    return (pick("STR", "gflops_fp32"), pick("AGI", "latency_ms"),
            pick("INT", "tokens_per_s"), pick("VIT", "copy_gbps"),
            pick("END", "sustained_gflops"))


def score_device(label, outcome):
    """Turn a suite outcome into a device result dict."""
    if outcome.failed_step is not None:
        return results.device_result(
            label, outcome.raw, outcome.training_race, None, None, None, None,
            {"step": outcome.failed_step, "message": outcome.error_message})
    stats = scoring.compute_stats(*metrics_from_raw(outcome.raw))
    score = scoring.compute_score(stats)
    tier = scoring.score_to_tier(score)
    str_gflops = outcome.raw["STR"]["gflops_fp32"]
    return results.device_result(
        label, outcome.raw, outcome.training_race, stats, score, tier,
        history.compare(str_gflops), None)


def _gpu_suite(config_, torch_mod, info, context_fn, steps, out, warnings):
    """Run the GPU suite; returns (device_result, gpu_name). Never raises
    Exception (KeyboardInterrupt still propagates)."""
    label = info.label
    try:
        ctx = context_fn(torch_mod, info)
    except Exception as exc:
        message = "%s: %s" % (type(exc).__name__, exc)
        warnings.append(
            "GPU initialisation failed on %s (%s); using CPU results"
            % (label, message))
        res = results.device_result(
            label, {}, None, None, None, None, None,
            {"step": "GPU initialisation", "message": message})
        return res, None
    gpu_name = getattr(ctx, "gpu_name", None)
    out("")
    out("Benchmarking %s%s" % (label, " (%s)" % gpu_name if gpu_name else ""))
    try:
        outcome = run_suite(torch_mod, ctx, config_.mode, steps, out)
        res = score_device(label, outcome)
    except Exception as exc:
        message = "%s: %s" % (type(exc).__name__, exc)
        warnings.append("GPU scoring failed on %s (%s); using CPU results"
                        % (label, message))
        return results.device_result(
            label, {}, None, None, None, None, None,
            {"step": "GPU scoring", "message": message}), gpu_name
    if outcome.failed_step is not None:
        warnings.append("GPU step '%s' failed on %s: %s; using CPU results"
                        % (outcome.failed_step, label, outcome.error_message))
    return res, gpu_name


def run_all(run_config, *, torch_mod, precheck, detect_fn=None,
            context_fn=None, cpu_context_fn=None, steps_fn=None, hw_fn=None,
            ram_fn=None, out=None, now_fn=None):
    """Run the whole benchmark and return the scrubbed result document."""
    detect_fn = detect_fn or devices.detect_backend
    context_fn = context_fn or devices.make_context
    cpu_context_fn = cpu_context_fn or devices.cpu_context
    steps_fn = steps_fn or default_steps
    hw_fn = hw_fn or hwinfo.collect
    ram_fn = ram_fn or hwinfo.total_ram_bytes
    out = out or display.say
    now_fn = now_fn or datetime.now

    _seed(torch_mod)
    steps = steps_fn()
    try:
        ram = ram_fn()
    except Exception:
        ram = None

    # Detect the GPU backend before any CPU work: torch_directml must be
    # imported before the first autograd backward pass, or DirectML training
    # fails with an internal device-index assertion.
    info = None
    if not run_config.cpu_only:
        try:
            info = detect_fn(torch_mod)
        except Exception:
            info = devices.BackendInfo(devices.LABEL_CPU, devices.KIND_CPU)

    out("")
    out("Benchmarking CPU")
    cpu_ctx = cpu_context_fn(torch_mod, ram)
    cpu_outcome = run_suite(torch_mod, cpu_ctx, run_config.mode, steps, out)
    if cpu_outcome.failed_step is not None:
        raise BenchmarkError("CPU step '%s' failed: %s" % (
            cpu_outcome.failed_step, cpu_outcome.error_message))
    cpu_res = score_device("CPU", cpu_outcome)

    warnings = list(getattr(precheck, "warnings", []) or [])
    gpu_res = None
    backend_label = devices.LABEL_CPU
    gpu_name = None
    if info is not None and info.label != devices.LABEL_CPU:
        backend_label = info.label
        gpu_res, gpu_name = _gpu_suite(
            run_config, torch_mod, info, context_fn, steps, out, warnings)

    primary = "gpu" if gpu_res and gpu_res["score"] is not None else "cpu"
    hardware = hw_fn(torch_mod, backend_label, gpu_name)
    doc = results.build_document(
        run_config.nickname, run_config.mode, warnings, hardware, cpu_res,
        gpu_res, primary, precheck, now_fn())
    return results.scrub_document(doc)
