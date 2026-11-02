"""The five RPG sub-benchmarks: STR, AGI, INT, VIT and END.

Every step has the signature fn(torch_mod, ctx, budget_s, mode, out) and
returns a dict that always holds budget_s and elapsed_s (measured from entry).
Data and weights are generated on the CPU with a seeded generator and then
moved to the target device; ctx.sync runs inside every timed region.
"""
import time
import warnings

from rpgbench import config
from rpgbench.timing import calibrate_inner, measure, median

MIB = 1024 * 1024
GIB = 1024 ** 3
MATMUL_LADDER = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096)


def _generator(torch_mod, offset=0):
    return torch_mod.Generator().manual_seed(config.SEED + offset)


def _finish(result, budget_s, start, torch_mod):
    result["threads"] = int(torch_mod.get_num_threads())
    result["budget_s"] = budget_s
    result["elapsed_s"] = time.perf_counter() - start
    return result


def _matrices(torch_mod, ctx, n):
    gen = _generator(torch_mod)
    a = torch_mod.randn(n, n, generator=gen).to(ctx.device)
    b = torch_mod.randn(n, n, generator=gen).to(ctx.device)
    return a, b


def run_str(torch_mod, ctx, budget_s, mode, out=None):
    """STR: square fp32 matmul throughput in GFLOPS."""
    start = time.perf_counter()
    target_batch = budget_s / 12.0
    chosen = MATMUL_LADDER[0]
    a = b = None
    with torch_mod.no_grad():
        for n in MATMUL_LADDER:
            ta, tb = _matrices(torch_mod, ctx, n)
            ctx.sync(torch_mod.matmul(ta, tb))
            begin = time.perf_counter()
            ctx.sync(torch_mod.matmul(ta, tb))
            single = time.perf_counter() - begin
            if a is not None and single > target_batch / 2.0:
                break
            a, b, chosen = ta, tb, n
            if single > target_batch / 2.0:
                break
        holder = {}

        def run_once():
            holder["c"] = torch_mod.matmul(a, b)

        def sync():
            ctx.sync(holder.get("c"))

        inner = calibrate_inner(run_once, sync, target_batch,
                                max_inner=4096)

        def run_batch():
            for _ in range(inner):
                run_once()
            sync()

        fp16_on = bool(ctx.supports_fp16)
        fp32_budget = budget_s * (0.85 if fp16_on else 1.0)
        m = measure(run_batch, budget_s=fp32_budget, start_time=start)
        flops = 2.0 * chosen ** 3 * inner
        samples = [flops / t / 1e9 for t in m.samples_s]

        gflops16 = None
        if fp16_on:
            try:
                gflops16 = _fp16_gflops(torch_mod, ctx, a, b, chosen,
                                        budget_s, start)
                note = "fp16 matmul measured"
            except Exception as exc:
                gflops16 = None
                note = "fp16 failed: %s" % type(exc).__name__
        elif ctx.kind == "cpu":
            note = "not attempted on CPU"
        else:
            note = "not supported on this backend"
    result = {
        "gflops_fp32": median(samples),
        "gflops_fp16": gflops16,
        "fp16_note": note,
        "matrix_n": chosen,
        "inner_iters": inner,
        "warmup_iters": m.warmup_iters,
        "samples": samples,
    }
    return _finish(result, budget_s, start, torch_mod)


def _fp16_gflops(torch_mod, ctx, a, b, n, budget_s, start):
    a16 = a.half()
    b16 = b.half()
    holder = {}

    def run_once():
        holder["c"] = torch_mod.matmul(a16, b16)

    def sync():
        ctx.sync(holder.get("c"))

    inner = calibrate_inner(run_once, sync, budget_s / 40.0, max_inner=4096)

    def batch():
        for _ in range(inner):
            holder["c"] = torch_mod.matmul(a16, b16)
        ctx.sync(holder["c"])

    m = measure(batch, budget_s=budget_s * 0.98, start_time=start,
                max_repeats=20)
    flops = 2.0 * n ** 3 * inner
    return median([flops / t / 1e9 for t in m.samples_s])


def run_agi(torch_mod, ctx, budget_s, mode, out=None):
    """AGI: small CNN, batch 1, mean per-call latency in milliseconds."""
    start = time.perf_counter()
    nn = torch_mod.nn
    torch_mod.manual_seed(config.SEED)
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64 * 8 * 8, 10))
    model = model.to(ctx.device).eval()
    batch_size = 1
    x = torch_mod.randn(batch_size, 3, 64, 64,
                        generator=_generator(torch_mod)).to(ctx.device)

    def run_once():
        ctx.sync(model(x))

    with torch_mod.no_grad():
        inner = calibrate_inner(run_once, lambda: None, budget_s / 20.0,
                                max_inner=100000)

        def run_batch():
            for _ in range(inner):
                run_once()

        m = measure(run_batch, budget_s=budget_s, start_time=start)
    samples = [t / inner * 1000.0 for t in m.samples_s]
    result = {
        "latency_ms": median(samples),
        "batch_size": batch_size,
        "inner_iters": inner,
        "warmup_iters": m.warmup_iters,
        "samples": samples,
        "model": "cnn 3x(conv3x3+relu+maxpool) + linear, input 3x64x64",
    }
    return _finish(result, budget_s, start, torch_mod)


VOCAB = 2048
D_MODEL = 256
LAYERS = 4
HEADS = 4
FF = 1024
PROMPT_LEN = 16
MIN_GEN = 16
MAX_GEN = 128


def run_int(torch_mod, ctx, budget_s, mode, out=None):
    """INT: greedy token generation from a random-init transformer."""
    start = time.perf_counter()
    nn = torch_mod.nn
    torch_mod.manual_seed(config.SEED)
    embed = nn.Embedding(VOCAB, D_MODEL)
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=HEADS, dim_feedforward=FF, dropout=0.0,
        batch_first=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoder = nn.TransformerEncoder(layer, num_layers=LAYERS,
                                        enable_nested_tensor=False)
    head = nn.Linear(D_MODEL, VOCAB)
    embed = embed.to(ctx.device).eval()
    encoder = encoder.to(ctx.device).eval()
    head = head.to(ctx.device).eval()
    prompt = torch_mod.randint(
        0, VOCAB, (1, PROMPT_LEN),
        generator=_generator(torch_mod)).to(ctx.device)
    masks = {}
    for length in range(PROMPT_LEN, PROMPT_LEN + MAX_GEN + 1):
        mask = torch_mod.full((length, length), float("-inf")).triu(1)
        masks[length] = mask.to(ctx.device)

    def generate(count):
        seq = prompt
        for _ in range(count):
            hidden = encoder(embed(seq), mask=masks[seq.shape[1]])
            nxt = head(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
            seq = torch_mod.cat([seq, nxt], dim=1)
        ctx.sync(seq)
        return seq

    with torch_mod.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generate(2)
        begin = time.perf_counter()
        generate(MIN_GEN)
        took = max(time.perf_counter() - begin, 1e-9)
        target_batch = budget_s / 12.0
        gen_tokens = int(MIN_GEN * target_batch / took)
        gen_tokens = max(MIN_GEN, min(MAX_GEN, gen_tokens))
        m = measure(lambda: generate(gen_tokens), budget_s=budget_s,
                    start_time=start)
    samples = [gen_tokens / t for t in m.samples_s]
    result = {
        "tokens_per_s": median(samples),
        "gen_tokens": gen_tokens,
        "prompt_len": PROMPT_LEN,
        "model": {"vocab": VOCAB, "d_model": D_MODEL, "layers": LAYERS,
                  "heads": HEADS},
        "warmup_iters": m.warmup_iters,
        "samples": samples,
    }
    return _finish(result, budget_s, start, torch_mod)


def run_vit(torch_mod, ctx, budget_s, mode, out=None):
    """VIT: memory copy bandwidth, then a capped allocation probe."""
    start = time.perf_counter()
    cap = int(ctx.alloc_cap_bytes)
    buf_mib = max(1, min(256, cap // 4 // MIB))
    base = torch_mod.randint(0, 256, (MIB,), dtype=torch_mod.uint8,
                             generator=_generator(torch_mod))
    src = base.repeat(buf_mib).to(ctx.device)
    dst = torch_mod.empty_like(src)
    nbytes = buf_mib * MIB

    def run_once():
        dst.copy_(src)

    def sync():
        ctx.sync(dst)

    with torch_mod.no_grad():
        inner = calibrate_inner(run_once, sync, budget_s / 20.0,
                                max_inner=4096)

        def run_batch():
            for _ in range(inner):
                run_once()
            sync()

        m = measure(run_batch, budget_s=budget_s * 0.85, start_time=start)
    samples = [2.0 * nbytes * inner / t / 1e9 for t in m.samples_s]
    del src, dst, base
    largest = _probe_allocations(torch_mod, ctx, cap, budget_s, start)
    result = {
        "copy_gbps": median(samples),
        "convention": "read+write counted (2 * bytes per copy)",
        "buffer_mb": buf_mib,
        "inner_iters": inner,
        "warmup_iters": m.warmup_iters,
        "samples": samples,
        "largest_alloc_gb": largest / GIB,
        "alloc_cap_gb": cap / GIB,
    }
    return _finish(result, budget_s, start, torch_mod)


def _probe_allocations(torch_mod, ctx, cap, budget_s, start):
    """Double the allocation size until OOM, the cap or the budget."""
    largest = 0
    size = min(128 * MIB, cap)
    while size > 0:
        if time.perf_counter() - start >= budget_s:
            break
        try:
            tensor = torch_mod.empty(size, dtype=torch_mod.uint8,
                                     device=ctx.device).fill_(1)
            ctx.sync(tensor)
            del tensor
        except (RuntimeError, MemoryError):
            break
        largest = size
        if size >= cap:
            break
        size = min(size * 2, cap)
    return largest


def run_end(torch_mod, ctx, budget_s, mode, out=None):
    """END: sustained matmul load; compare first and last windows."""
    start = time.perf_counter()
    duration = min(config.END_DURATION_S[mode], max(0.5, budget_s - 0.5))
    window = min(config.END_WINDOW_S[mode], duration / 5.0)
    n = 512
    with torch_mod.no_grad():
        while True:
            a, b = _matrices(torch_mod, ctx, n)
            ctx.sync(torch_mod.matmul(a, b))
            begin = time.perf_counter()
            ctx.sync(torch_mod.matmul(a, b))
            single = time.perf_counter() - begin
            if single >= 0.02 or n >= 4096:
                break
            n *= 2
        flops = 2.0 * n ** 3
        ends = []
        loop_start = time.perf_counter()
        while True:
            c = torch_mod.matmul(a, b)
            ctx.sync(c)
            now = time.perf_counter() - loop_start
            ends.append(now)
            if now >= duration:
                break
    total = ends[-1]
    first_k = max(1, sum(1 for t in ends if t <= window))
    first = first_k * flops / ends[first_k - 1] / 1e9
    cutoff = total - window
    idx = next((i for i, t in enumerate(ends) if t > cutoff), len(ends) - 1)
    span_start = ends[idx - 1] if idx > 0 else 0.0
    last_span = max(total - span_start, 1e-9)
    last = (len(ends) - idx) * flops / last_span / 1e9
    result = {
        "first_window_gflops": first,
        "last_window_gflops": last,
        "sustained_gflops": len(ends) * flops / total / 1e9,
        "throttle_pct": (first - last) / first * 100.0 if first > 0 else 0.0,
        "duration_s": total,
        "window_s": window,
        "matrix_n": n,
        "ops": len(ends),
    }
    return _finish(result, budget_s, start, torch_mod)


STEPS = (
    ("STR", run_str),
    ("AGI", run_agi),
    ("INT", run_int),
    ("VIT", run_vit),
    ("END", run_end),
)
