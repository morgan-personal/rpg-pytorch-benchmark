import copy
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

import pytest

from rpgbench import results

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "sample_result.json"
NOW = datetime(2026, 11, 1, 9, 30, 5)


@pytest.fixture
def doc():
    with open(FIXTURE, encoding="utf-8") as handle:
        return json.load(handle)


def test_fixture_is_valid(doc):
    assert results.validate_document(doc) == []


def test_fixture_required_fields(doc):
    assert isinstance(doc["schema_version"], int)
    assert isinstance(doc["nickname"], str)
    datetime.fromisoformat(doc["timestamp"])
    assert doc["mode"] in ("quick", "full")
    assert all(isinstance(w, str) for w in doc["warnings"])
    for key in ("cpu_model", "cpu_cores_logical", "cpu_cores_physical",
                "ram_gb", "gpu_name", "torch_version", "backend",
                "python_version", "os"):
        assert key in doc["hardware"]
    cpu = doc["results"]["cpu"]
    assert cpu["device_label"] == "CPU"
    assert doc["results"]["gpu"] is None
    assert doc["hardware"]["backend"] == "CPU"
    assert cpu["score"] > 0
    assert cpu["comparison"]
    assert set(cpu["stats"]) == {"STR", "AGI", "INT", "VIT", "END"}


def test_fixture_median_invariant(doc):
    raw = doc["results"]["cpu"]["raw"]
    pairs = [("STR", "gflops_fp32"), ("AGI", "latency_ms"),
             ("INT", "tokens_per_s"), ("VIT", "copy_gbps")]
    for step, key in pairs:
        assert raw[step]["warmup_iters"] >= 1
        assert len(raw[step]["samples"]) >= 3
        assert raw[step][key] == pytest.approx(
            statistics.median(raw[step]["samples"]), rel=1e-9)
    assert raw["STR"]["gflops_fp16"] is None
    assert raw["AGI"]["batch_size"] <= 8
    assert raw["VIT"]["largest_alloc_gb"] <= raw["VIT"]["alloc_cap_gb"]


def test_fixture_budget_keys(doc):
    raw = doc["results"]["cpu"]["raw"]
    for step in ("STR", "AGI", "INT", "VIT", "END", "RACE"):
        assert raw[step]["elapsed_s"] <= 2 * raw[step]["budget_s"] + 3


def test_built_document_is_valid(doc):
    cpu = doc["results"]["cpu"]
    built = results.build_document(
        "smoke", "quick", ["w"], doc["hardware"], cpu, None, "cpu",
        {"on_battery": False, "heavy_apps": [], "notes": []}, NOW)
    assert results.validate_document(built) == []
    assert built["schema_version"] == 1
    datetime.fromisoformat(built["timestamp"])
    assert built["precheck"]["on_battery"] is False


def test_validate_catches_problems(doc):
    bad = copy.deepcopy(doc)
    bad["results"]["cpu"]["raw"]["STR"]["gflops_fp32"] = 1.0
    bad["results"]["cpu"]["tier"] = "Legend"
    del bad["hardware"]["os"]
    bad["results"]["cpu"]["raw"]["AGI"]["samples"] = [1.0, 2.0]
    del bad["results"]["cpu"]["raw"]["END"]["budget_s"]
    text = " ".join(results.validate_document(bad))
    for needle in ("gflops_fp32", "tier", "hardware.os", "AGI.samples",
                   "END.budget_s"):
        assert needle in text
    assert results.validate_document({}) != []
    assert results.validate_document([]) != []


def test_scrub_removes_paths_and_tokens(doc):
    doc["nickname"] = "bobpc fan"
    doc["warnings"] = [
        r"failed at C:\Users\alice\proj\x.py on BobPC",
        "also d:/data/stuff.txt here",
        r"unc \\server\share\file ok",
    ]
    doc["hardware"]["cpu_model"] = "Intel ALICE inside"
    scrubbed = results.scrub_document(doc, ["bobpc", "alice", "ab"])
    text = json.dumps(scrubbed)
    assert "alice" not in text.lower()
    assert "bobpc" not in text.lower().replace("bobpc fan", "")
    assert not re.search(r"[A-Za-z]:\\", text)
    assert "<path>" in text and "<redacted>" in text
    assert scrubbed["nickname"] == "bobpc fan"
    assert doc["warnings"][0].startswith("failed at C:")


def test_scrub_ignores_short_tokens_and_keeps_enums(doc):
    scrubbed = results.scrub_document(doc, ["ab", "CPU", "quick", "Workhorse"])
    assert scrubbed["results"]["cpu"]["device_label"] == "CPU"
    assert scrubbed["mode"] == "quick"
    assert scrubbed["results"]["cpu"]["tier"] == "Workhorse"
    assert results.validate_document(scrubbed) == []


def test_scrub_default_tokens_do_not_break_document(doc):
    assert results.validate_document(results.scrub_document(doc)) == []


def test_timestamp_stem_pattern():
    stem = results.timestamp_stem("smoke", NOW)
    assert stem == "smoke_20261101-093005"
    assert re.fullmatch(r"smoke_\d{8}-\d{6}", stem)


def test_reserve_stem_suffixes(tmp_path):
    base = "a_20261101-093005"
    assert results.reserve_stem(tmp_path, base) == base
    (tmp_path / (base + ".json")).write_text("{}")
    assert results.reserve_stem(tmp_path, base) == base + "-2"
    (tmp_path / (base + "-2.png")).write_bytes(b"x")
    assert results.reserve_stem(tmp_path, base) == base + "-3"


def _png_card(doc_, path):
    Path(path).write_bytes(b"png")


def test_save_run_writes_json_and_png(tmp_path, doc):
    res = results.save_run(doc, tmp_path / "results", _png_card, "smoke", NOW)
    assert res.json_name == "smoke_20261101-093005.json"
    assert res.png_name == "smoke_20261101-093005.png"
    assert (tmp_path / "results" / res.json_name).exists()
    assert (tmp_path / "results" / res.png_name).exists()
    loaded = json.loads((tmp_path / "results" / res.json_name).read_text(
        encoding="utf-8"))
    assert loaded["nickname"] == "smoke"


def test_save_run_collision_does_not_overwrite(tmp_path, doc):
    folder = tmp_path / "results"
    first = results.save_run(doc, folder, _png_card, "smoke", NOW)
    before = (folder / first.json_name).read_bytes()
    second = results.save_run(doc, folder, _png_card, "smoke", NOW)
    third = results.save_run(doc, folder, _png_card, "smoke", NOW)
    assert second.json_name.endswith("-2.json")
    assert second.png_name.endswith("-2.png")
    assert third.json_name.endswith("-3.json")
    assert (folder / first.json_name).read_bytes() == before
    assert len(list(folder.glob("*.json"))) == 3
    assert len(list(folder.glob("*.png"))) == 3


def test_save_run_png_only_collision(tmp_path, doc):
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "smoke_20261101-093005.png").write_bytes(b"old")
    res = results.save_run(doc, folder, _png_card, "smoke", NOW)
    assert res.json_name == "smoke_20261101-093005-2.json"
    assert (folder / "smoke_20261101-093005.png").read_bytes() == b"old"


def test_save_run_card_failure_still_writes_json(tmp_path, doc):
    def broken(doc_, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("boom")

    res = results.save_run(doc, tmp_path, broken, "smoke", NOW)
    assert res.png_name is None
    assert not list(tmp_path.glob("*.png"))
    assert res.warnings_added
    saved = json.loads((tmp_path / res.json_name).read_text(encoding="utf-8"))
    assert any(w.startswith("Character card not rendered: RuntimeError")
               for w in saved["warnings"])


def test_save_run_names_are_bare(tmp_path, doc):
    res = results.save_run(doc, tmp_path, _png_card, "smoke", NOW)
    for name in (res.json_name, res.png_name):
        assert "/" not in name and "\\" not in name and ":" not in name


def test_save_run_derives_file_part_from_nickname(tmp_path, doc):
    doc["nickname"] = "Ana/Maria: <x>"
    res = results.save_run(doc, tmp_path, _png_card, now=NOW)
    assert re.fullmatch(r"[A-Za-z0-9_-]+_20261101-093005\.json",
                        res.json_name)
