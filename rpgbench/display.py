"""ASCII-only console output and the end-of-run summary."""
import sys

STAT_ORDER = ("STR", "AGI", "INT", "VIT", "END")


def say(text="", end="\n", stream=None):
    """Print text as ASCII (unencodable characters become '?') and flush."""
    stream = stream if stream is not None else sys.stdout
    data = (str(text) + end).encode("ascii", "replace").decode("ascii")
    stream.write(data)
    stream.flush()


def step_line(device_label, step, summary, elapsed_s):
    return "  [%s] %-4s done in %5.1f s - %s" % (
        device_label, step, float(elapsed_s), summary)


def race_summary(race):
    """'reached 90% in 4.2 s after 340 steps' or 'DNF - best accuracy 0.87'."""
    if not race:
        return "n/a"
    if race.get("reached") and race.get("time_to_target_s") is not None:
        pct = int(round(float(race.get("target_accuracy", 0.9)) * 100))
        return "reached %d%% in %.1f s after %d steps" % (
            pct, float(race["time_to_target_s"]),
            int(race.get("steps_completed", 0)))
    return "DNF - best accuracy %.2f" % float(race.get("best_accuracy", 0.0))


def throttle_summary(raw):
    pct = ((raw or {}).get("END") or {}).get("throttle_pct")
    if not isinstance(pct, (int, float)) or pct != pct:
        return "n/a"
    if pct <= 0:
        return "no throttle"
    return "%.0f%% throttle" % pct


def step_summary(step, res):
    """One-line description of a finished step result."""
    try:
        if step == "STR":
            return "%.1f GFLOPS fp32" % res["gflops_fp32"]
        if step == "AGI":
            return "%.2f ms per inference" % res["latency_ms"]
        if step == "INT":
            return "%.1f tokens/s" % res["tokens_per_s"]
        if step == "VIT":
            return "%.1f GB/s copy, largest alloc %.2f GB" % (
                res["copy_gbps"], res["largest_alloc_gb"])
        if step == "END":
            return "%.1f GFLOPS sustained, %s" % (
                res["sustained_gflops"], throttle_summary({"END": res}))
        if step == "RACE":
            return race_summary(res)
    except (KeyError, TypeError, ValueError):
        pass
    return "done"


def _device_block(key, doc, primary):
    res = doc["results"].get(key)
    if not res:
        return []
    mark = " (primary)" if key == primary else ""
    if res.get("score") is None:
        return ["[%s]%s unscored - see warnings" % (res["device_label"], mark)]
    stats = res["stats"]
    lines = [
        "[%s]%s  score %.1f - %s" % (
            res["device_label"], mark, res["score"], res["tier"]),
        "  " + "  ".join("%s %.1f" % (n, stats[n]) for n in STAT_ORDER),
        "  Raw power: %s" % res["comparison"],
        "  Training Race: %s" % race_summary(res.get("training_race")),
        "  Sustained load: %s" % throttle_summary(res.get("raw")),
    ]
    return lines


def format_summary(doc, json_rel, png_rel):
    """Build the full end-of-run text (callers print it through say)."""
    primary = doc.get("primary", "cpu")
    lines = ["", "=" * 60,
             "RESULTS for %s (%s mode)" % (doc.get("nickname", "player"),
                                           doc.get("mode", "?")),
             "=" * 60]
    for key in ("cpu", "gpu"):
        block = _device_block(key, doc, primary)
        if block:
            lines.extend(block)
    main = doc["results"].get(primary) or doc["results"]["cpu"]
    lines.append("")
    lines.append("Overall: %s, %s" % (
        "%.1f" % main["score"] if main.get("score") is not None else "n/a",
        main.get("tier")))
    warnings = doc.get("warnings") or []
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend("  - %s" % w for w in warnings)
    lines.append("")
    lines.append("Results saved:")
    lines.append("  %s" % json_rel)
    lines.append("  %s" % (png_rel if png_rel else "(character card not written)"))
    lines.append("Send both files to your instructor.")
    return "\n".join(lines)


def print_summary(doc, json_rel, png_rel, out=say):
    text = format_summary(doc, json_rel, png_rel)
    for line in text.split("\n"):
        out(line)
