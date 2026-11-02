"""Training Race: train a small MLP on synthetic data until it hits a target."""
import sys
import time

from rpgbench import config

DIMS = 32
CLASSES = 10
N_TRAIN = 8192
N_VAL = 2048
BATCH = 128
HIDDEN = 128
LR = 1e-3
EVAL_EVERY = 10
CENTER_SCALE = 1.0
NOISE_STD = 0.6
CLUSTERS_PER_CLASS = 32
BAR_WIDTH = 30
TTY_REDRAW_S = 0.2


def make_dataset(seed, n_train=N_TRAIN, n_val=N_VAL, dims=DIMS,
                 classes=CLASSES, torch_mod=None):
    """Gaussian class blobs, generated on the CPU from a seeded generator."""
    if torch_mod is None:
        import torch as torch_mod
    gen = torch_mod.Generator().manual_seed(seed)
    blobs = classes * CLUSTERS_PER_CLASS
    centers = torch_mod.randn(blobs, dims, generator=gen) * CENTER_SCALE

    def draw(count):
        blob = torch_mod.randint(0, blobs, (count,), generator=gen)
        noise = torch_mod.randn(count, dims, generator=gen) * NOISE_STD
        return centers[blob] + noise, blob % classes

    x_train, y_train = draw(n_train)
    x_val, y_val = draw(n_val)
    return x_train, y_train, x_val, y_val


class ProgressBar:
    """ASCII progress bar: carriage-return redraw on a tty, lines otherwise."""

    def __init__(self, out, tty, width=BAR_WIDTH, clock=time.perf_counter):
        self._out = out
        self._tty = bool(tty)
        self._width = width
        self._clock = clock
        self._last_draw = None
        self._last_bucket = -1
        self._line = ""
        self._printed = None

    def _format(self, fraction, step, accuracy):
        fraction = min(1.0, max(0.0, fraction))
        filled = int(round(fraction * self._width))
        bar = "#" * filled + "-" * (self._width - filled)
        return "[%s] %3d%% step %d acc %.3f" % (
            bar, int(fraction * 100), step, accuracy)

    def update(self, fraction, step, accuracy, force=False):
        self._line = self._format(fraction, step, accuracy)
        if self._tty:
            now = self._clock()
            if (force or self._last_draw is None
                    or now - self._last_draw >= TTY_REDRAW_S):
                self._out("\r" + self._line, end="")
                self._last_draw = now
            return
        bucket = int(min(1.0, max(0.0, fraction)) * 10)
        if force or bucket > self._last_bucket:
            self._out(self._line)
            self._printed = self._line
            self._last_bucket = max(self._last_bucket, bucket)

    def finish(self, fraction, step, accuracy):
        line = self._format(fraction, step, accuracy)
        if self._tty:
            self._out("\r" + line, end="")
            self._out("")
        elif line != self._printed:
            self._out(line)
            self._printed = line


def _noop_out(text="", end="\n"):
    return None


def run_race(torch_mod, ctx, budget_s, mode, out=None, stream=None):
    """Run the race and return the Training Race result dict."""
    start = time.perf_counter()
    out = out or _noop_out
    stream = stream if stream is not None else sys.stdout
    try:
        tty = bool(stream.isatty())
    except Exception:
        tty = False
    seed = config.SEED
    target = config.RACE_TARGET_ACCURACY
    nn = torch_mod.nn

    x_train, y_train, x_val, y_val = make_dataset(seed, torch_mod=torch_mod)
    x_train, y_train = x_train.to(ctx.device), y_train.to(ctx.device)
    x_val, y_val = x_val.to(ctx.device), y_val.to(ctx.device)
    torch_mod.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(DIMS, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
        nn.Linear(HIDDEN, CLASSES)).to(ctx.device)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    shuffle_gen = torch_mod.Generator().manual_seed(seed)
    batches_per_epoch = N_TRAIN // BATCH
    chance = 1.0 / CLASSES

    def evaluate():
        with torch_mod.no_grad():
            pred = model(x_val).argmax(dim=1)
            correct = (pred == y_val).float().mean()
            ctx.sync(correct)
            return float(correct.item())

    def progress(best, begin):
        acc_part = (best - chance) / max(target - chance, 1e-9)
        time_part = (time.perf_counter() - begin) / max(budget_s, 1e-9)
        return min(1.0, max(0.0, acc_part, time_part))

    bar = ProgressBar(out, tty)
    best = evaluate()
    steps = 0
    reached = best >= target
    time_to_target = 0.0 if reached else None
    train_start = time.perf_counter()
    bar.update(0.0, 0, best, force=True)
    last_eval_step = 0
    done = reached
    while not done:
        perm = torch_mod.randperm(N_TRAIN, generator=shuffle_gen).to(
            ctx.device)
        xs = x_train[perm]
        ys = y_train[perm]
        for b in range(batches_per_epoch):
            lo = b * BATCH
            optimizer.zero_grad()
            loss = loss_fn(model(xs[lo:lo + BATCH]), ys[lo:lo + BATCH])
            loss.backward()
            optimizer.step()
            steps += 1
            if steps % EVAL_EVERY == 0:
                ctx.sync(loss)
                acc = evaluate()
                last_eval_step = steps
                best = max(best, acc)
                bar.update(progress(best, train_start), steps, best)
                if acc >= target:
                    reached = True
                    time_to_target = time.perf_counter() - train_start
                    done = True
                    break
            if time.perf_counter() - train_start >= budget_s:
                done = True
                break
    if last_eval_step != steps:
        best = max(best, evaluate())
        if best >= target and not reached:
            reached = True
            time_to_target = time.perf_counter() - train_start
    bar.finish(1.0 if reached else progress(best, train_start), steps, best)
    return {
        "target_accuracy": target,
        "reached": bool(reached),
        "time_to_target_s": time_to_target if reached else None,
        "best_accuracy": best,
        "steps_completed": steps,
        "epochs_completed": steps // batches_per_epoch,
        "seed": seed,
        "budget_s": budget_s,
        "elapsed_s": time.perf_counter() - start,
    }
