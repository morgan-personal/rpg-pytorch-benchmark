import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_script(*args):
    return subprocess.run([sys.executable, *args], cwd=str(ROOT),
                          capture_output=True, timeout=120)


def test_bench_help():
    res = run_script("bench.py", "--help")
    assert res.returncode == 0
    text = res.stdout.decode("ascii", "replace")
    for flag in ("--quick", "--name", "--cpu-only"):
        assert flag in text


def test_leaderboard_help():
    res = run_script("leaderboard.py", "--help")
    assert res.returncode == 0
    text = res.stdout.decode("ascii", "replace")
    assert "--include-quick" in text and "--output" in text


def test_bench_rejects_unknown_option():
    res = run_script("bench.py", "--bogus")
    assert res.returncode == 2
    assert b"Traceback" not in res.stderr
