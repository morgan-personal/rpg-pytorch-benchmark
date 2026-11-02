import copy
import io
import json
from pathlib import Path

import pytest

from rpgbench import display

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_result.json"


@pytest.fixture
def doc():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_say_replaces_non_ascii():
    buf = io.StringIO()
    display.say("café ✓", stream=buf)
    assert buf.getvalue() == "caf? ?\n"


def test_say_end_parameter():
    buf = io.StringIO()
    display.say("a", end="", stream=buf)
    assert buf.getvalue() == "a"


def test_summary_contents(doc):
    text = display.format_summary(doc, "results\\smoke.json",
                                  "results\\smoke.png")
    assert text.isascii()
    for needle in ("STR 96.7", "AGI 194.7", "INT 232.8", "VIT 65.5",
                   "END 90.5", "136.0", "Workhorse", "CM-5",
                   "reached 90% in 0.9 s after 910 steps",
                   "Running on battery", "results\\smoke.json",
                   "results\\smoke.png", "6% throttle"):
        assert needle in text


def test_summary_dnf_and_clamped_throttle(doc):
    cpu = doc["results"]["cpu"]
    cpu["training_race"].update(reached=False, time_to_target_s=None,
                                best_accuracy=0.87)
    cpu["raw"]["END"]["throttle_pct"] = -8.0
    text = display.format_summary(doc, "j", "p")
    assert "DNF - best accuracy 0.87" in text
    assert "no throttle" in text
    assert "-8" not in text


def test_summary_gpu_unscored_and_primary(doc):
    gpu = copy.deepcopy(doc["results"]["cpu"])
    gpu.update(device_label="DirectML", score=None, stats=None, tier=None,
               comparison=None, training_race=None)
    doc["results"]["gpu"] = gpu
    doc["warnings"].append("GPU step 'INT' failed on DirectML: boom")
    text = display.format_summary(doc, "j", None)
    assert "[DirectML] unscored - see warnings" in text
    assert "GPU step 'INT' failed" in text
    assert "(character card not written)" in text
    assert "[CPU] (primary)" in text


def test_summary_non_ascii_goes_through_say(doc):
    doc["nickname"] = "José"
    buf = io.StringIO()
    display.print_summary(
        doc, "j", "p", out=lambda line: display.say(line, stream=buf))
    assert buf.getvalue().isascii()
    assert "Jos?" in buf.getvalue()


def test_step_summaries():
    assert "GFLOPS" in display.step_summary("STR", {"gflops_fp32": 5.0})
    assert display.step_summary("STR", {}) == "done"
    assert "tokens/s" in display.step_summary("INT", {"tokens_per_s": 3.0})
    line = display.step_line("CPU", "STR", "x", 3.14159)
    assert "CPU" in line and "3.1" in line
