"""Result document assembly, privacy scrubbing, validation and saving."""
import copy
import getpass
import json
import math
import os
import platform
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rpgbench import BENCH_VERSION, SCHEMA_VERSION
from rpgbench import nickname as nickname_mod

TIER_NAMES = ("Potato", "Toaster", "Workhorse", "Gaming Rig",
              "Mini Supercomputer")
HARDWARE_KEYS = ("cpu_model", "cpu_cores_logical", "cpu_cores_physical",
                 "ram_gb", "gpu_name", "torch_version", "backend",
                 "python_version", "os")
STAT_KEYS = ("STR", "AGI", "INT", "VIT", "END")
MEDIAN_FIELDS = (("STR", "gflops_fp32"), ("AGI", "latency_ms"),
                 ("INT", "tokens_per_s"), ("VIT", "copy_gbps"))
RACE_KEYS = ("target_accuracy", "reached", "time_to_target_s",
             "best_accuracy", "seed")

REDACTED = "<redacted>"
PATH_MARK = "<path>"
_DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
_UNC_PATH = re.compile(r"\\\\[^\s\"']+")
# Whole-string values that must survive scrubbing unchanged.
_PROTECTED = frozenset(
    ("quick", "full", "cpu", "gpu", "CPU", "AMD ROCm", "NVIDIA CUDA",
     "Apple MPS", "DirectML") + TIER_NAMES)


@dataclass
class SaveResult:
    json_name: str
    png_name: object = None
    warnings_added: list = field(default_factory=list)


def device_result(label, raw, training_race, stats, score, tier, comparison,
                  error=None):
    return {
        "device_label": label,
        "stats": stats,
        "score": score,
        "tier": tier,
        "comparison": comparison,
        "training_race": training_race,
        "raw": raw,
        "error": error,
    }


def _precheck_dict(precheck):
    if precheck is None:
        return {"on_battery": None, "heavy_apps": [], "notes": []}
    get = (precheck.get if isinstance(precheck, dict)
           else lambda k, d=None: getattr(precheck, k, d))
    return {
        "on_battery": get("on_battery", None),
        "heavy_apps": list(get("heavy_apps", []) or []),
        "notes": list(get("notes", []) or []),
    }


def build_document(nickname, mode, warnings, hardware, cpu_result,
                   gpu_result, primary, precheck, now):
    """Assemble the schema_version 1 document (before scrubbing)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "bench_version": BENCH_VERSION,
        "nickname": nickname,
        "timestamp": now.astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "primary": primary,
        "warnings": list(warnings),
        "precheck": _precheck_dict(precheck),
        "hardware": dict(hardware),
        "results": {"cpu": cpu_result, "gpu": gpu_result},
    }


def _identity_tokens():
    candidates = [os.environ.get("COMPUTERNAME"), os.environ.get("USERNAME"),
                  os.environ.get("USER")]
    try:
        candidates.append(platform.node())
    except Exception:
        pass
    try:
        candidates.append(getpass.getuser())
    except Exception:
        pass
    tokens = []
    for token in candidates:
        if token and len(token) >= 3 and token not in tokens:
            tokens.append(token)
    return tokens


def scrub_text(text, tokens):
    """Replace paths and identity tokens in one string."""
    if text in _PROTECTED:
        return text
    text = _UNC_PATH.sub(PATH_MARK, _DRIVE_PATH.sub(PATH_MARK, text))
    for token in tokens:
        text = re.sub(re.escape(token), REDACTED, text, flags=re.IGNORECASE)
    return text


def _scrub_value(value, tokens):
    if isinstance(value, str):
        return scrub_text(value, tokens)
    if isinstance(value, dict):
        return {k: _scrub_value(v, tokens) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(v, tokens) for v in value]
    return value


def scrub_document(doc, forbidden_tokens=None):
    """Return a copy without drive/UNC paths or hostname/username tokens.

    The top-level nickname is the user's own chosen text and is kept.
    """
    if forbidden_tokens is None:
        tokens = _identity_tokens()
    else:
        tokens = [t for t in forbidden_tokens if t and len(t) >= 3]
    out = {}
    for key, value in doc.items():
        if key == "nickname":
            out[key] = value
        else:
            out[key] = _scrub_value(copy.deepcopy(value), tokens)
    return out


def _finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _check_device(name, res, problems, require_scored):
    if not isinstance(res, dict):
        problems.append("results.%s is not an object" % name)
        return
    scored = res.get("score") is not None
    if require_scored or scored:
        stats = res.get("stats")
        if not isinstance(stats, dict):
            problems.append("results.%s.stats missing" % name)
        else:
            for key in STAT_KEYS:
                value = stats.get(key)
                if not _finite_number(value) or value < 0:
                    problems.append("results.%s.stats.%s invalid" % (name, key))
        score = res.get("score")
        if not _finite_number(score) or score <= 0:
            problems.append("results.%s.score invalid" % name)
        if res.get("tier") not in TIER_NAMES:
            problems.append("results.%s.tier invalid" % name)
        comparison = res.get("comparison")
        if not isinstance(comparison, str) or not comparison:
            problems.append("results.%s.comparison invalid" % name)
        race = res.get("training_race")
        if not isinstance(race, dict):
            problems.append("results.%s.training_race missing" % name)
        else:
            for key in RACE_KEYS:
                if key not in race:
                    problems.append("results.%s.training_race.%s missing"
                                    % (name, key))
            if "steps_completed" not in race and "epochs_completed" not in race:
                problems.append("results.%s.training_race has no progress"
                                % name)
    if not isinstance(res.get("device_label"), str):
        problems.append("results.%s.device_label missing" % name)
    raw = res.get("raw")
    if not isinstance(raw, dict):
        problems.append("results.%s.raw missing" % name)
        return
    for step in ("STR", "AGI", "INT", "VIT", "END", "RACE"):
        entry = raw.get(step)
        if entry is None:
            if require_scored or scored:
                problems.append("results.%s.raw.%s missing" % (name, step))
            continue
        for key in ("budget_s", "elapsed_s"):
            if not _finite_number(entry.get(key)):
                problems.append("results.%s.raw.%s.%s missing"
                                % (name, step, key))
    for step, key in MEDIAN_FIELDS:
        entry = raw.get(step)
        if not isinstance(entry, dict):
            continue
        samples = entry.get("samples")
        if (not isinstance(samples, list) or len(samples) < 3
                or not all(_finite_number(s) for s in samples)):
            problems.append("results.%s.raw.%s.samples invalid" % (name, step))
            continue
        warm = entry.get("warmup_iters")
        if not isinstance(warm, int) or warm < 1:
            problems.append("results.%s.raw.%s.warmup_iters invalid"
                            % (name, step))
        value = entry.get(key)
        if not _finite_number(value) or not math.isclose(
                value, statistics.median(samples), rel_tol=1e-9):
            problems.append("results.%s.raw.%s.%s is not the median"
                            % (name, step, key))
    agi = raw.get("AGI")
    if isinstance(agi, dict):
        size = agi.get("batch_size")
        if not isinstance(size, int) or not 1 <= size <= 8:
            problems.append("results.%s.raw.AGI.batch_size invalid" % name)
    vit = raw.get("VIT")
    if isinstance(vit, dict):
        big, cap = vit.get("largest_alloc_gb"), vit.get("alloc_cap_gb")
        if not (_finite_number(big) and _finite_number(cap) and big <= cap):
            problems.append("results.%s.raw.VIT allocation fields invalid"
                            % name)
    if name == "cpu" and isinstance(raw.get("STR"), dict):
        if raw["STR"].get("gflops_fp16") is not None:
            problems.append("results.cpu.raw.STR.gflops_fp16 must be null")


def validate_document(doc):
    """Return a list of problems (empty when the document is valid)."""
    problems = []
    if not isinstance(doc, dict):
        return ["document is not an object"]
    if not isinstance(doc.get("schema_version"), int) or isinstance(
            doc.get("schema_version"), bool):
        problems.append("schema_version is not an integer")
    if not isinstance(doc.get("nickname"), str):
        problems.append("nickname is not a string")
    stamp = doc.get("timestamp")
    try:
        datetime.fromisoformat(stamp)
    except Exception:
        problems.append("timestamp is not ISO 8601")
    if doc.get("mode") not in ("quick", "full"):
        problems.append("mode is not quick or full")
    warnings = doc.get("warnings")
    if not isinstance(warnings, list) or not all(
            isinstance(w, str) for w in warnings):
        problems.append("warnings is not a list of strings")
    hardware = doc.get("hardware")
    if not isinstance(hardware, dict):
        problems.append("hardware is not an object")
    else:
        for key in HARDWARE_KEYS:
            if key not in hardware:
                problems.append("hardware.%s missing" % key)
    results = doc.get("results")
    if not isinstance(results, dict) or "cpu" not in results:
        problems.append("results.cpu missing")
        return problems
    cpu = results["cpu"]
    _check_device("cpu", cpu, problems, True)
    if isinstance(cpu, dict) and cpu.get("device_label") != "CPU":
        problems.append("results.cpu.device_label is not CPU")
    gpu = results.get("gpu")
    if gpu is not None:
        _check_device("gpu", gpu, problems, False)
    return problems


def timestamp_stem(file_part, now):
    return "%s_%s" % (file_part, now.strftime("%Y%m%d-%H%M%S"))


def _taken(directory, stem):
    return ((directory / (stem + ".json")).exists()
            or (directory / (stem + ".png")).exists())


def reserve_stem(directory, base):
    """First of base, base-2, base-3 ... with no .json or .png present."""
    directory = Path(directory)
    stem = base
    counter = 1
    while _taken(directory, stem):
        counter += 1
        stem = "%s-%d" % (base, counter)
    return stem


def save_run(doc, results_dir, card_fn, file_part=None, now=None):
    """Write the card PNG then the JSON; never overwrite an earlier run."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if file_part is None:
        file_part = nickname_mod.file_part(doc.get("nickname", ""))
    base = timestamp_stem(file_part, now or datetime.now())
    stem = reserve_stem(results_dir, base)
    added = []
    png_name = stem + ".png"
    png_path = results_dir / png_name
    try:
        card_fn(doc, png_path)
    except Exception as exc:
        text = "Character card not rendered: %s: %s" % (
            type(exc).__name__, exc)
        added.append(scrub_text(text, _identity_tokens()))
        doc.setdefault("warnings", []).append(added[-1])
        png_name = None
        try:
            if png_path.exists():
                png_path.unlink()
        except OSError:
            pass
    for _ in range(1000):
        json_name = stem + ".json"
        try:
            handle = open(results_dir / json_name, "x", encoding="utf-8")
        except FileExistsError:
            new_stem = reserve_stem(results_dir, base)
            if png_name and (results_dir / png_name).exists():
                os.replace(results_dir / png_name,
                           results_dir / (new_stem + ".png"))
                png_name = new_stem + ".png"
            stem = new_stem
            continue
        with handle:
            json.dump(doc, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return SaveResult(json_name, png_name, added)
    raise OSError("could not reserve a results file name")
