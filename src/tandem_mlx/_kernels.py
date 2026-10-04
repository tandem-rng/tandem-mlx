"""The Metal kernels: thin MLX wrappers over the functions of tandem.metal."""

from importlib.resources import files

import mlx.core as mx
import numpy as np

SOURCE = (files(__package__) / "tandem.metal").read_text()

# Fast math would replace the IEEE division and square root of the normals and exponentials.
_OPTIONS = {"math_mode": "safe"}
_THREADS = 256

# Kinds of tandem::fill, as in tandem.metal.
BITS32, BITS64, UNIFORM32, BELOW32, BELOW32_64, BELOW64, NORMAL32, EXPONENTIAL32, NORMAL32_ODD = range(9)
_WIDTH = {BITS64: 64, BELOW64: 64}

# Elements per launch. Below 2^31 the kernel's element offsets fit in 32 bits, and an even
# count keeps the normal pairs of a split fill where the whole fill has them.
WINDOW = 2**30

_fill = mx.fast.metal_kernel(
    name="tandem_fill",
    input_names=["key", "params"],
    output_names=["out"],
    source="""
    tandem::FillParams p = tandem::unpack(key, params);
    tandem::fill<KIND, K>((device uint *)out, p, thread_position_in_grid.x);
    """,
    header=SOURCE,
    compile_options=_OPTIONS,
)

_derive = mx.fast.metal_kernel(
    name="tandem_derive",
    input_names=["key", "spec", "domain"],
    output_names=["out"],
    source="""
    uint i = thread_position_in_grid.x;
    uint4 k = uint4(key[0], key[1], key[2], key[3]);
    uint4 q = uint4(spec[4 * i], spec[4 * i + 1], spec[4 * i + 2], spec[4 * i + 3]);
    tandem::State s = tandem::F_keyed(k, ulong(q.x) | (ulong(q.y) << 32), domain[0], q.z);
    uint4 r = q.w ? s.h : s.o;
    out[4 * i] = r.x, out[4 * i + 1] = r.y, out[4 * i + 2] = r.z, out[4 * i + 3] = r.w;
    """,
    header=SOURCE,
    compile_options=_OPTIONS,
)

def _words64(x):
    return [x & 0xFFFFFFFF, (x >> 32) & 0xFFFFFFFF]


def _launch(key, kind, aligned, n, K, dtype, range_, lo):
    w = _WIDTH.get(kind, 32)
    per = 128 // w
    d0 = aligned // w
    b0, skip = divmod(d0, per)
    g_first = (b0 >> 3) // K
    g_last = (((d0 + n - 1) // per) >> 3) // K
    params = [*_words64(g_first), b0 - g_first * 8 * K, skip, n, *_words64(d0), *_words64(range_), *_words64(lo)]
    threads = 8 * (g_last - g_first + 1)
    return _fill(
        inputs=[key, mx.array(np.array(params, np.uint32))],
        template=[("KIND", kind), ("K", K)],
        grid=(threads, 1, 1),
        threadgroup=(min(_THREADS, threads), 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[dtype],
    )[0]


def fill(key, kind, aligned, n, K, dtype, range_=0, lo=0):
    """`n` elements of `kind` from the aligned bit position, as a flat array of `dtype`."""
    w = _WIDTH.get(kind, 32)
    parts = [
        _launch(key, kind, aligned + w * s, min(WINDOW, n - s), K, dtype, range_, lo % 2**64)
        for s in range(0, n, WINDOW)
    ]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts)


def derive(key, spec, domain):
    """Rows of `half` of F(key, counter, domain, aux) for the uint32 rows (counter lo, hi, aux, half)."""
    m = spec.shape[0]
    return _derive(
        inputs=[key, mx.array(spec), mx.array(np.array([domain], np.uint32))],
        grid=(m, 1, 1),
        threadgroup=(min(_THREADS, m), 1, 1),
        output_shapes=[(m, 4)],
        output_dtypes=[mx.uint32],
    )[0]
