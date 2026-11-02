"""Hardware-history comparison: how does your GFLOPS stack up over the years."""
import math

# (name, approximate fp GFLOPS), ascending. Figures are rough on purpose.
REFERENCE_MACHINES = (
    ("Apollo Guidance Computer", 0.00004),
    ("#1 supercomputer of 1976 (Cray-1)", 0.16),
    ("#1 supercomputer of 1993 (CM-5)", 60.0),
    ("#1 supercomputer of 1997 (ASCI Red)", 1068.0),
    ("#1 supercomputer of 2002 (Earth Simulator)", 35860.0),
    ("#1 supercomputer of 2008 (Roadrunner)", 1.03e6),
    ("#1 supercomputer of 2022 (Frontier)", 1.1e9),
)


def _fmt(number):
    if number < 10:
        return "%.1f" % number
    if number < 1e12:
        return "{:,.0f}".format(number)
    return "%.1e" % number


def compare(gflops):
    """Return an ASCII sentence comparing gflops with a historic machine."""
    try:
        value = float(gflops)
    except (TypeError, ValueError):
        value = float("nan")
    first_name, first_gflops = REFERENCE_MACHINES[0]
    if math.isnan(value) or value <= 0:
        return "too slow to register next to the %s" % first_name
    if math.isinf(value):
        return "faster than the %s" % REFERENCE_MACHINES[-1][0]
    best = None
    for name, ref in REFERENCE_MACHINES:
        if ref <= value:
            best = (name, ref)
    if best is None:
        return "about 1/%s of the %s" % (_fmt(first_gflops / value), first_name)
    name, ref = best
    return "about %sx the %s" % (_fmt(value / ref), name)
