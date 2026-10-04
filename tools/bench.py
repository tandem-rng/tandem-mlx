"""GiB/s of output on the default GPU at 2^24 and 2^26 elements, against mx.random, as a markdown
table. A run issues ten calls back to back and evaluates them together. The table gives the best
of five runs after a warm-up. Run it on a quiet machine.
"""

import time

import mlx.core as mx

import tandem_mlx as tm

KEY = tm.key(42)
MKEY = mx.random.key(42)

# (label, bytes per element, tandem_mlx, mx.random or None)
ROWS = [
    ("uniform float32", 4, lambda n: tm.uniform(KEY, n), lambda n: mx.random.uniform(shape=(n,), key=MKEY)),
    ("bits uint32", 4, lambda n: tm.bits(KEY, n), None),
    ("randint int32 in [0, 1000)", 4, lambda n: tm.randint(KEY, n, 0, 1000), lambda n: mx.random.randint(0, 1000, (n,), key=MKEY)),
    ("normal float32", 4, lambda n: tm.normal(KEY, n), lambda n: mx.random.normal((n,), key=MKEY)),
    ("normal float32, odd start", 4, lambda n: tm.normal(KEY, n, position=32), None),
    ("exponential float32", 4, lambda n: tm.exponential(KEY, n), None),
]


def rate(f, n, nbytes, calls=10, runs=5):
    mx.eval(f(n))
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        mx.eval(*[f(n) for _ in range(calls)])
        best = min(best, time.perf_counter() - t0)
    return calls * n * nbytes / best / 2**30


print(f"{mx.device_info()['device_name']}, mlx {mx.__version__}")
print("| draw | log2 n | tandem_mlx | mx.random |")
print("|---|---|---|---|")
for log2n in (24, 26):
    for label, nbytes, ours, theirs in ROWS:
        n = 2**log2n
        cells = [f"{rate(f, n, nbytes):.0f}" if f else "-" for f in (ours, theirs)]
        print(f"| {label} | {log2n} | " + " | ".join(cells) + " |")
