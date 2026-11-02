import json
import re

import pytest

import leaderboard
import rpgbench

STATS = {"STR": 100.0, "AGI": 90.0, "INT": 80.0, "VIT": 70.0, "END": 60.0}


def device(score, label="CPU", race=None):
    return {
        "device_label": label,
        "stats": dict(STATS),
        "score": score,
        "tier": "Workhorse",
        "comparison": "about 2.0x the Cray-1",
        "training_race": race or {"reached": True, "time_to_target_s": 3.2,
                                  "best_accuracy": 0.91},
        "raw": {},
        "error": None,
    }


def make_doc(nickname, cpu_score, gpu_score=None, mode="full",
             cpu_model="Test CPU", gpu_label="AMD ROCm"):
    return {
        "schema_version": 1,
        "nickname": nickname,
        "timestamp": "2026-11-01T10:00:00+00:00",
        "mode": mode,
        "hardware": {"cpu_model": cpu_model,
                     "gpu_name": "Test GPU" if gpu_score else None},
        "results": {
            "cpu": device(cpu_score),
            "gpu": device(gpu_score, gpu_label) if gpu_score else None,
        },
    }


def write(folder, name, doc):
    path = folder / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def run(folder, out, *extra):
    return leaderboard.main([str(folder), "-o", str(out), *extra])


def test_schema_version_matches_package():
    assert rpgbench.SCHEMA_VERSION in leaderboard.SUPPORTED_SCHEMA_VERSIONS


def test_ranking_boards_and_fields(tmp_path, capsys):
    write(tmp_path, "a.json", make_doc("Ana", 100.0, 400.0))
    write(tmp_path, "b.json", make_doc("Ben", 200.0))
    write(tmp_path, "c.json", make_doc("Cy", 50.0))
    out = tmp_path / "out.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "CPU board" in page and "GPU board" in page
    cpu_part = page.split("GPU board")[0]
    assert cpu_part.index("Ben") < cpu_part.index("Ana") < cpu_part.index("Cy")
    assert "4.0x" in cpu_part
    gpu_part = page.split("GPU board")[1]
    assert "Ana" in gpu_part and "Ben" not in gpu_part
    assert "1.0x" in gpu_part
    for needle in ("Workhorse", "AMD ROCm", "100.0", "90.0", "80.0", "70.0",
                   "60.0", "3.2 s", "Test CPU", "Test GPU"):
        assert needle in page
    assert re.search(r'<td class="num">1</td>', cpu_part)
    assert re.search(r'<td class="num">3</td>', cpu_part)


def test_default_output_in_cwd(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    write(data, "a.json", make_doc("Ana", 10.0))
    monkeypatch.chdir(tmp_path)
    assert leaderboard.main([str(data)]) == 0
    assert (tmp_path / "leaderboard.html").exists()


def test_gpu_empty_state_and_single_entry_spread(tmp_path):
    write(tmp_path, "a.json", make_doc("Ana", 10.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "No GPU results yet." in page
    assert "1.0x" in page.split("GPU board")[0]


def test_spread_helper():
    assert leaderboard.spread([]) == "n/a"
    assert leaderboard.spread([{"score": 5.0}]) == "1.0x"
    assert leaderboard.spread([{"score": 9.0}, {"score": 3.0}]) == "3.0x"
    assert leaderboard.spread([{"score": 9.0}, {"score": 0.0}]) == "n/a"


def test_empty_folder(tmp_path):
    folder = tmp_path / "empty"
    folder.mkdir()
    out = tmp_path / "o.html"
    assert run(folder, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "No CPU results yet." in page and "No GPU results yet." in page


def test_missing_folder(tmp_path, capsys):
    code = leaderboard.main([str(tmp_path / "nope"), "-o",
                             str(tmp_path / "o.html")])
    captured = capsys.readouterr()
    assert code != 0
    assert "error" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "o.html").exists()


def test_bad_files_skipped_with_warning(tmp_path, capsys):
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    write(tmp_path, "nokeys.json", {"schema_version": 1, "nickname": "x"})
    bad_version = make_doc("Old", 10.0)
    bad_version["schema_version"] = 99
    write(tmp_path, "oldver.json", bad_version)
    write(tmp_path, "list.json", [1, 2, 3])
    (tmp_path / "binary.json").write_bytes(b"\xff\xfe\x00bad")
    write(tmp_path, "good.json", make_doc("Good", 10.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    err = capsys.readouterr().err
    for name in ("broken.json", "nokeys.json", "oldver.json", "list.json",
                 "binary.json"):
        assert "warning: skipping %s:" % name in err
    assert str(tmp_path) not in err
    page = out.read_text(encoding="utf-8")
    assert "Good" in page
    assert "5 file(s) skipped" in page


def test_quick_excluded_by_default_and_included_on_flag(tmp_path):
    write(tmp_path, "q.json", make_doc("Quicky", 10.0, mode="quick"))
    write(tmp_path, "f.json", make_doc("Fully", 20.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "Quicky" not in page and "Fully" in page
    assert "1 quick-mode run(s) left out" in page
    assert run(tmp_path, out, "--include-quick") == 0
    page = out.read_text(encoding="utf-8")
    assert "Quicky" in page and "Fully" in page


def test_dedupe_keeps_highest_per_nickname(tmp_path):
    write(tmp_path, "1.json", make_doc("Ana", 10.0, 300.0))
    write(tmp_path, "2.json", make_doc("ana", 90.0, 100.0))
    write(tmp_path, "3.json", make_doc("Ben", 50.0))
    docs, _ = leaderboard.load_documents(tmp_path, lambda t: None)
    boards, _ = leaderboard.build_boards(docs)
    assert [(e["nickname"], e["score"]) for e in boards["cpu"]] == [
        ("ana", 90.0), ("Ben", 50.0)]
    assert [(e["nickname"], e["score"]) for e in boards["gpu"]] == [
        ("Ana", 300.0)]


def test_gpu_without_finite_score_ignored(tmp_path):
    doc = make_doc("Ana", 10.0, 5.0)
    doc["results"]["gpu"]["score"] = None
    write(tmp_path, "a.json", doc)
    docs, _ = leaderboard.load_documents(tmp_path, lambda t: None)
    boards, _ = leaderboard.build_boards(docs)
    assert boards["gpu"] == []
    assert len(boards["cpu"]) == 1


def test_html_escaping(tmp_path):
    doc = make_doc("<script>alert(1)</script>", 10.0,
                   cpu_model='<b>"Fast" & \'loud\'</b>')
    write(tmp_path, "x.json", doc)
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<b>" not in page
    assert "&lt;b&gt;&quot;Fast&quot; &amp; &#x27;loud&#x27;&lt;/b&gt;" in page


def test_long_nickname_truncated(tmp_path):
    write(tmp_path, "x.json", make_doc("N" * 100, 10.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert "N" * 32 in page and "N" * 33 not in page


def test_page_is_self_contained(tmp_path):
    write(tmp_path, "x.json", make_doc("Ana", 10.0, 20.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    page = out.read_text(encoding="utf-8")
    assert not re.search(
        r"""(src|href)\s*=\s*["']?\s*(https?:)?//|@import|<script[^>]*\ssrc=""",
        page)
    assert "<script" not in page.lower()
    assert "<style>" in page


def test_non_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    write(sub, "deep.json", make_doc("Deep", 10.0))
    out = tmp_path / "o.html"
    assert run(tmp_path, out) == 0
    assert "Deep" not in out.read_text(encoding="utf-8")


def test_unwritable_output_returns_error(tmp_path, capsys):
    write(tmp_path, "x.json", make_doc("Ana", 10.0))
    code = leaderboard.main([str(tmp_path), "-o",
                             str(tmp_path / "no" / "such" / "o.html")])
    assert code == 2
    assert "Traceback" not in capsys.readouterr().err
