"""Character-card PNG (matplotlib, non-interactive Agg backend)."""
import math
import textwrap

STAT_ORDER = ("STR", "AGI", "INT", "VIT", "END")

BG = "#14171f"
PANEL = "#1d2230"
FG = "#e8e6df"
ACCENT = "#f2b84b"
MUTED = "#8b93a7"


def _ascii(text):
    return str(text).encode("ascii", "replace").decode("ascii")


def race_line(race):
    """One-line Training Race summary (time or DNF)."""
    if not race:
        return "Training Race: n/a"
    if race.get("reached") and race.get("time_to_target_s") is not None:
        pct = int(round(float(race.get("target_accuracy", 0.9)) * 100))
        return "Training Race: reached %d%% in %.1f s" % (
            pct, float(race["time_to_target_s"]))
    return "Training Race: DNF (best accuracy %.2f)" % float(
        race.get("best_accuracy", 0.0))


def throttle_line(raw):
    end = (raw or {}).get("END") or {}
    pct = end.get("throttle_pct")
    if not isinstance(pct, (int, float)) or not math.isfinite(pct):
        return "Sustained load: n/a"
    if pct <= 0:
        return "Sustained load: no throttle"
    return "Sustained load: %.0f%% throttle" % pct


def render_card(doc, png_path):
    """Draw the character card for the primary device and save it as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = doc.get("primary", "cpu")
    results = doc["results"]
    main = results.get(primary) or results["cpu"]
    stats = main["stats"]
    values = [float(stats[name]) for name in STAT_ORDER]
    rmax = max(150.0, max(values) * 1.15)

    fig = plt.figure(figsize=(8, 10), dpi=100, facecolor=BG)
    try:
        fig.text(0.5, 0.945, _ascii(doc.get("nickname", "player")),
                 ha="center", va="center", fontsize=30, color=ACCENT,
                 fontweight="bold")
        fig.text(0.5, 0.895,
                 _ascii("%s  |  %s" % (main["device_label"], main["tier"])),
                 ha="center", va="center", fontsize=20, color=FG)
        fig.text(0.5, 0.848, "Overall score  %.1f" % float(main["score"]),
                 ha="center", va="center", fontsize=18, color=ACCENT)

        ax = fig.add_axes([0.16, 0.33, 0.68, 0.44], projection="polar",
                          facecolor=PANEL)
        angles = [2 * math.pi * i / len(STAT_ORDER)
                  for i in range(len(STAT_ORDER))]
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles + angles[:1], values + values[:1], color=ACCENT,
                linewidth=2.5)
        ax.fill(angles + angles[:1], values + values[:1], color=ACCENT,
                alpha=0.35)
        ax.set_xticks(angles)
        ax.set_xticklabels(STAT_ORDER, color=FG, fontsize=17,
                           fontweight="bold")
        ax.set_ylim(0, rmax)
        ax.set_yticks([rmax / 4, rmax / 2, rmax * 3 / 4])
        ax.set_yticklabels([])
        ax.grid(color=MUTED, alpha=0.5)
        ax.spines["polar"].set_color(MUTED)

        stat_text = "   ".join("%s %.0f" % (n, v)
                               for n, v in zip(STAT_ORDER, values))
        fig.text(0.5, 0.285, stat_text, ha="center", va="center",
                 fontsize=14, color=FG, family="monospace")
        fig.text(0.5, 0.235, _ascii(race_line(main.get("training_race"))),
                 ha="center", va="center", fontsize=15, color=FG)
        fig.text(0.5, 0.195, _ascii(throttle_line(main.get("raw"))),
                 ha="center", va="center", fontsize=13, color=MUTED)
        comparison = _ascii("Raw power: " + str(main.get("comparison", "")))
        fig.text(0.5, 0.135, textwrap.fill(comparison, 58), ha="center",
                 va="center", fontsize=13, color=FG)
        if primary == "gpu" and results.get("cpu"):
            cpu = results["cpu"]
            fig.text(0.5, 0.07,
                     _ascii("CPU score %.1f  (%s)" % (
                         float(cpu["score"]), cpu["tier"])),
                     ha="center", va="center", fontsize=13, color=MUTED)
        fig.savefig(str(png_path), dpi=100, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
