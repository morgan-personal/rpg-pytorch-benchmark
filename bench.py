"""RPG-style PyTorch hardware benchmark: python bench.py [--quick] [--name X]"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NO_TORCH = 2
EXIT_INTERRUPTED = 130


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench.py",
        description=(
            "Measure your computer as an RPG character: STR (matrix math), "
            "AGI (small-model latency), INT (text generation), VIT (memory) "
            "and END (sustained load), plus a Training Race. A full run "
            "takes about 3-5 minutes on a CPU-only machine (up to about 8 "
            "minutes with a GPU). Results are written to the results/ "
            "folder next to this script."))
    parser.add_argument(
        "--quick", action="store_true",
        help="short smoke-test run (about 30 seconds, less accurate)")
    parser.add_argument(
        "--name", metavar="NICKNAME", default=None,
        help="your nickname for the character card and results file "
             "(default: ask, or 'player' when not interactive)")
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="skip GPU detection and only benchmark the CPU")
    return parser


def _import_torch(say):
    try:
        import torch
    except Exception as exc:
        say("PyTorch could not be imported (%s)." % type(exc).__name__)
        say("Install it first - see the install sections in README.md.")
        return None
    return torch


def _run(args):
    from rpgbench import card, display, nickname, precheck, results, runner

    say = display.say
    shown, file_part = nickname.resolve_nickname(
        args.name, sys.stdin, input, say)
    mode = "quick" if args.quick else "full"
    say("")
    say("=== RPG PyTorch Benchmark (%s mode) ===" % mode)
    say("Player: %s" % shown)
    say("Tip: close heavy apps and plug in your charger for a fair score.")
    pre = precheck.run_precheck()
    for warning in pre.warnings:
        say("WARNING: %s" % warning)
    for note in pre.notes:
        say("Note: %s" % note)

    torch = _import_torch(say)
    if torch is None:
        return EXIT_NO_TORCH

    run_config = runner.RunConfig(mode=mode, cpu_only=args.cpu_only,
                                  nickname=shown, file_part=file_part)
    try:
        doc = runner.run_all(run_config, torch_mod=torch, precheck=pre,
                             out=say)
    except runner.BenchmarkError as exc:
        say("")
        say("Benchmark failed: %s" % exc)
        return EXIT_FAILED

    try:
        saved = results.save_run(doc, RESULTS_DIR, card.render_card,
                                 file_part=file_part)
    except OSError as exc:
        say("")
        say("Could not write results (%s): %s" % (
            type(exc).__name__, exc.strerror or "see folder permissions"))
        return EXIT_FAILED
    json_rel = os.path.join("results", saved.json_name)
    png_rel = os.path.join("results", saved.png_name) if saved.png_name else None
    display.print_summary(doc, json_rel, png_rel)
    return EXIT_OK


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print()
        print("Interrupted - no results written.")
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
