import json
import struct
import sys
from datetime import datetime
from pathlib import Path

import pytest

from rpgbench import card, results

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_result.json"
NOW = datetime(2026, 11, 1, 9, 30, 5)


@pytest.fixture
def doc():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_matplotlib_unavailable_json_still_written(tmp_path, doc, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    res = results.save_run(doc, tmp_path, card.render_card, "smoke", NOW)
    assert res.png_name is None
    assert (tmp_path / res.json_name).exists()
    saved = json.loads((tmp_path / res.json_name).read_text(encoding="utf-8"))
    assert any("Character card not rendered" in w for w in saved["warnings"])


def test_race_and_throttle_lines():
    assert "DNF" in card.race_line({"reached": False, "best_accuracy": 0.87})
    assert "2.5 s" in card.race_line(
        {"reached": True, "time_to_target_s": 2.5, "target_accuracy": 0.9})
    assert "no throttle" in card.throttle_line({"END": {"throttle_pct": -4}})
    assert "12%" in card.throttle_line({"END": {"throttle_pct": 12.0}})
    assert card.throttle_line(None) == "Sustained load: n/a"


def test_real_render_dimensions(tmp_path, doc):
    pytest.importorskip("matplotlib")
    target = tmp_path / "card.png"
    doc["nickname"] = "José the Brave"
    card.render_card(doc, target)
    header = target.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", header[16:24])
    assert width >= 600 and height >= 600


def test_real_render_gpu_primary(tmp_path, doc):
    pytest.importorskip("matplotlib")
    gpu = json.loads(json.dumps(doc["results"]["cpu"]))
    gpu["device_label"] = "AMD ROCm"
    doc["results"]["gpu"] = gpu
    doc["primary"] = "gpu"
    target = tmp_path / "gpu.png"
    card.render_card(doc, target)
    assert target.exists()
