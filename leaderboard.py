"""Turn a folder of benchmark result JSON files into one static HTML page.

Standard library only. Usage:

    python leaderboard.py FOLDER [-o leaderboard.html] [--include-quick]
"""
import argparse
import html
import json
import math
import sys
from pathlib import Path

# Keep in step with rpgbench.SCHEMA_VERSION (a unit test checks this).
SUPPORTED_SCHEMA_VERSIONS = (1,)
MAX_FILE_BYTES = 5 * 1024 * 1024
NICK_DISPLAY_LEN = 32
STAT_NAMES = ("STR", "AGI", "INT", "VIT", "END")

CSS = """
body { background: #14171f; color: #e8e6df; font-family: Segoe UI, Arial,
  sans-serif; margin: 0; padding: 24px; }
h1 { color: #f2b84b; margin: 0 0 4px 0; }
h2 { color: #f2b84b; margin-top: 32px; }
.meta, .spread { color: #8b93a7; font-size: 0.95em; margin: 4px 0; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; }
th, td { border-bottom: 1px solid #2a3144; padding: 6px 8px; text-align: left;
  vertical-align: top; }
th { color: #8b93a7; font-weight: 600; font-size: 0.85em; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
td.name { word-break: break-word; overflow-wrap: anywhere; max-width: 260px; }
td.hw { word-break: break-word; overflow-wrap: anywhere; color: #8b93a7;
  font-size: 0.85em; max-width: 260px; }
.empty { color: #8b93a7; font-style: italic; padding: 12px 0; }
"""


def _is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def validate_entry(doc):
    """Return None when the document is usable, else a short reason."""
    if not isinstance(doc, dict):
        return "not a JSON object"
    version = doc.get("schema_version")
    if isinstance(version, bool) or version not in SUPPORTED_SCHEMA_VERSIONS:
        return "unsupported schema_version %s" % _short(version)
    if not isinstance(doc.get("nickname"), str):
        return "missing nickname"
    if doc.get("mode") not in ("quick", "full"):
        return "missing or invalid mode"
    if not isinstance(doc.get("hardware"), dict):
        return "missing hardware"
    results = doc.get("results")
    if not isinstance(results, dict) or not isinstance(
            results.get("cpu"), dict):
        return "missing results.cpu"
    cpu = results["cpu"]
    if not _is_number(cpu.get("score")):
        return "missing results.cpu.score"
    if not isinstance(cpu.get("stats"), dict):
        return "missing results.cpu.stats"
    if not isinstance(cpu.get("tier"), str):
        return "missing results.cpu.tier"
    return None


def _short(value):
    text = repr(value)
    return text if len(text) <= 24 else text[:21] + "..."


def load_documents(folder, err):
    """Read every *.json in folder (non-recursive). Returns (docs, skipped)."""
    docs = []
    skipped = 0
    for path in sorted(Path(folder).glob("*.json")):
        if not path.is_file():
            continue
        name = path.name
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                reason = "file too large"
                doc = None
            else:
                with open(path, "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
                reason = validate_entry(doc)
        except OSError as exc:
            doc, reason = None, "cannot read file (%s)" % type(exc).__name__
        except ValueError as exc:
            doc, reason = None, "malformed JSON (%s)" % type(exc).__name__
        except RecursionError:
            doc, reason = None, "malformed JSON (too deeply nested)"
        if reason is not None:
            skipped += 1
            err("warning: skipping %s: %s" % (name, reason))
            continue
        docs.append(doc)
    return docs, skipped


def _stat_value(stats, key):
    value = stats.get(key) if isinstance(stats, dict) else None
    return float(value) if _is_number(value) else None


def _race_text(race):
    if not isinstance(race, dict):
        return "n/a"
    if race.get("reached") is True and _is_number(
            race.get("time_to_target_s")):
        return "%.1f s" % race["time_to_target_s"]
    best = race.get("best_accuracy")
    if _is_number(best):
        return "DNF (best %.2f)" % best
    return "DNF"


def _device_entry(doc, device, model_text):
    res = doc["results"].get(device)
    if not isinstance(res, dict) or not _is_number(res.get("score")):
        return None
    stats = res.get("stats")
    label = res.get("device_label")
    return {
        "nickname": doc["nickname"].strip(),
        "score": float(res["score"]),
        "tier": str(res.get("tier")),
        "device_label": label if isinstance(label, str) else device.upper(),
        "stats": {k: _stat_value(stats, k) for k in STAT_NAMES},
        "race": _race_text(res.get("training_race")),
        "model": model_text,
    }


def _text_or_empty(value):
    return value if isinstance(value, str) else ""


def build_boards(docs, include_quick=False):
    """Return ({"cpu": [...], "gpu": [...]}, quick_excluded_count)."""
    excluded = 0
    best = {"cpu": {}, "gpu": {}}
    for doc in docs:
        if doc["mode"] == "quick" and not include_quick:
            excluded += 1
            continue
        hardware = doc["hardware"]
        for device, model in (
                ("cpu", _text_or_empty(hardware.get("cpu_model"))),
                ("gpu", _text_or_empty(hardware.get("gpu_name")))):
            entry = _device_entry(doc, device, model)
            if entry is None:
                continue
            key = entry["nickname"].casefold()
            current = best[device].get(key)
            if current is None or entry["score"] > current["score"]:
                best[device][key] = entry
    boards = {}
    for device, mapping in best.items():
        ranked = sorted(mapping.values(),
                        key=lambda e: (-e["score"], e["nickname"].casefold()))
        for index, entry in enumerate(ranked, start=1):
            entry["rank"] = index
        boards[device] = ranked
    return boards, excluded


def spread(entries):
    """Fastest/slowest score ratio as text."""
    if not entries:
        return "n/a"
    if len(entries) == 1:
        return "1.0x"
    top, bottom = entries[0]["score"], entries[-1]["score"]
    if not (math.isfinite(top) and math.isfinite(bottom)) or bottom <= 0:
        return "n/a"
    return "%.1fx" % (top / bottom)


def _esc(value):
    return html.escape(str(value), quote=True)


def _fmt(value):
    return "-" if value is None else "%.1f" % value


def _render_board(title, entries, empty_text):
    parts = ["<h2>%s</h2>" % _esc(title)]
    if not entries:
        parts.append('<p class="empty">%s</p>' % _esc(empty_text))
        return "\n".join(parts)
    parts.append('<p class="spread">Fastest vs slowest: %s (%d entries)</p>'
                 % (_esc(spread(entries)), len(entries)))
    head = ["Rank", "Nickname", "Score", "Tier", "Device"]
    head += list(STAT_NAMES) + ["Training Race", "Hardware"]
    parts.append("<table>")
    parts.append("<tr>%s</tr>" % "".join(
        "<th>%s</th>" % _esc(h) for h in head))
    for entry in entries:
        nick = entry["nickname"][:NICK_DISPLAY_LEN]
        cells = [
            '<td class="num">%s</td>' % _esc(entry["rank"]),
            '<td class="name">%s</td>' % _esc(nick),
            '<td class="num">%s</td>' % _esc("%.1f" % entry["score"]),
            "<td>%s</td>" % _esc(entry["tier"]),
            "<td>%s</td>" % _esc(entry["device_label"]),
        ]
        for key in STAT_NAMES:
            cells.append('<td class="num">%s</td>'
                         % _esc(_fmt(entry["stats"][key])))
        cells.append("<td>%s</td>" % _esc(entry["race"]))
        cells.append('<td class="hw">%s</td>' % _esc(entry["model"]))
        parts.append("<tr>%s</tr>" % "".join(cells))
    parts.append("</table>")
    return "\n".join(parts)


def render_html(boards, meta):
    """Render the whole self-contained page as a string."""
    notes = []
    if meta.get("quick_excluded"):
        notes.append("%d quick-mode run(s) left out (use --include-quick "
                     "to include them)." % meta["quick_excluded"])
    elif meta.get("include_quick"):
        notes.append("Quick-mode runs are included.")
    if meta.get("skipped"):
        notes.append("%d file(s) skipped because they could not be read "
                     "as valid results." % meta["skipped"])
    if not notes:
        notes.append("Showing each nickname's best run per board.")
    body = [
        "<h1>RPG Hardware Leaderboard</h1>",
        "".join('<p class="meta">%s</p>' % _esc(n) for n in notes),
        _render_board("CPU board", boards.get("cpu", []),
                      "No CPU results yet."),
        _render_board("GPU board", boards.get("gpu", []),
                      "No GPU results yet."),
    ]
    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<title>RPG Hardware Leaderboard</title>\n"
        "<style>%s</style>\n</head>\n<body>\n%s\n</body>\n</html>\n"
        % (CSS, "\n".join(body)))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="leaderboard.py",
        description="Build a static leaderboard.html from a folder of "
                    "benchmark result .json files.")
    parser.add_argument("folder", help="folder holding the result .json files")
    parser.add_argument("-o", "--output", default="leaderboard.html",
                        help="output HTML path (default: leaderboard.html "
                             "in the current directory)")
    parser.add_argument("--include-quick", action="store_true",
                        help="include quick-mode runs (left out by default)")
    args = parser.parse_args(argv)

    def err(text):
        print(text, file=sys.stderr)

    folder = Path(args.folder)
    if not folder.is_dir():
        err("error: folder not found: %s" % args.folder)
        return 2
    try:
        docs, skipped = load_documents(folder, err)
    except OSError as exc:
        err("error: cannot read folder (%s)" % type(exc).__name__)
        return 2
    boards, excluded = build_boards(docs, args.include_quick)
    page = render_html(boards, {
        "quick_excluded": excluded,
        "skipped": skipped,
        "include_quick": args.include_quick,
    })
    try:
        Path(args.output).write_text(page, encoding="utf-8")
    except OSError as exc:
        err("error: cannot write %s (%s)" % (args.output, type(exc).__name__))
        return 2
    print("Wrote %s (%d CPU, %d GPU entries)" % (
        args.output, len(boards["cpu"]), len(boards["gpu"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
