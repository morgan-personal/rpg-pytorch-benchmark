import io

import pytest

torch = pytest.importorskip("torch")

from rpgbench import devices, race  # noqa: E402


@pytest.fixture(scope="module")
def ctx():
    return devices.cpu_context(torch, 8 * 1024 ** 3)


def test_same_seed_identical_tensors():
    first = race.make_dataset(7, n_train=64, n_val=32)
    second = race.make_dataset(7, n_train=64, n_val=32)
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_different_seed_different_tensors():
    first = race.make_dataset(7, n_train=64, n_val=32)
    second = race.make_dataset(8, n_train=64, n_val=32)
    assert not torch.equal(first[0], second[0])


def test_dataset_shapes():
    x_train, y_train, x_val, y_val = race.make_dataset(1, 100, 50)
    assert tuple(x_train.shape) == (100, 32)
    assert tuple(x_val.shape) == (50, 32)
    assert int(y_train.max()) < 10 and int(y_train.min()) >= 0


def _collector():
    buf = io.StringIO()

    def out(text="", end="\n"):
        buf.write(text + end)

    return buf, out


def test_race_keys_and_progress_output(ctx):
    buf, out = _collector()
    res = race.run_race(torch, ctx, 1.0, "quick", out, stream=io.StringIO())
    for key in ("target_accuracy", "reached", "time_to_target_s",
                "best_accuracy", "steps_completed", "epochs_completed",
                "seed", "budget_s", "elapsed_s"):
        assert key in res
    assert res["budget_s"] == 1.0
    assert (res["time_to_target_s"] is None) == (not res["reached"])
    text = buf.getvalue()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) > 1
    assert text.isascii()
    assert "\r" not in text
    assert text.endswith("\n")
    assert "%" in lines[-1] and "acc" in lines[-1] and "step" in lines[-1]


def test_race_not_reached_with_tiny_budget(ctx):
    res = race.run_race(torch, ctx, 0.05, "quick", None,
                        stream=io.StringIO())
    assert res["reached"] is False
    assert res["time_to_target_s"] is None
    assert 0.0 <= res["best_accuracy"] <= 1.0


def test_progress_bar_tty_uses_carriage_return():
    buf, out = _collector()
    bar = race.ProgressBar(out, tty=True, clock=iter(range(100)).__next__)
    bar.update(0.1, 10, 0.2)
    bar.update(0.5, 50, 0.5)
    bar.finish(1.0, 90, 0.9)
    text = buf.getvalue()
    assert "\r" in text
    assert text.endswith("\n")
    assert text.isascii()


def test_progress_bar_non_tty_throttles_to_milestones():
    buf, out = _collector()
    bar = race.ProgressBar(out, tty=False)
    for i in range(1000):
        bar.update(i / 1000.0, i, 0.5)
    bar.finish(1.0, 1000, 0.9)
    assert len(buf.getvalue().splitlines()) <= 13


def test_dataset_dtypes():
    """Features are float32 and labels are int64, for both splits."""
    x_train, y_train, x_val, y_val = race.make_dataset(3, n_train=64, n_val=32)
    assert x_train.dtype == torch.float32
    assert x_val.dtype == torch.float32
    assert y_train.dtype == torch.int64
    assert y_val.dtype == torch.int64
