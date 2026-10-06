"""The Metal kernels: thin MLX wrappers over the fills of tandem.metal, which is tandem-metal's
shader source, vendored unchanged."""

from importlib.resources import files

import mlx.core as mx
import numpy as np

SOURCE = (files(__package__) / "tandem.metal").read_text()
# MLX writes its own kernels around a header, so the header leaves out tandem.metal's kernels.
HEADER = "#define TANDEM_NO_KERNELS\n" + SOURCE

# Fast math would replace the IEEE division and square root of the normals and exponentials.
_OPTIONS = {"math_mode": "safe"}
# tandem.metal's THREADS. The odd normal fill needs whole threadgroups of it.
_THREADS = 256

BITS32, BITS64, UNIFORM32, EXPONENTIAL32, BELOW32, BELOW32_64, BELOW64, NORMAL32, NORMAL32_ODD = range(9)
_WIDTH = {BITS64: 64, BELOW64: 64}

# KIND picks the element type of tandem::fill, in the order of the kinds above.
_fill = mx.fast.metal_kernel(
    name="tandem_fill",
    input_names=["key", "params"],
    output_names=["out"],
    source="""
    tandem::Params p = *(const device tandem::Params *)params;
    p.key = uint4(key[0], key[1], key[2], key[3]);
    uint t = thread_position_in_grid.x;
    device uchar *o = (device uchar *)out;
    if (KIND <= 1) tandem::fill<tandem::U32>(p, o, t);
    else if (KIND == 2) tandem::fill<tandem::F32>(p, o, t);
    else if (KIND == 3) tandem::fill<tandem::Exponential32>(p, o, t);
    else if (KIND == 4) tandem::fill<tandem::Below32>(p, o, t);
    else if (KIND == 5) tandem::fill<tandem::Wide32>(p, o, t);
    else if (KIND == 6) tandem::fill<tandem::Below64>(p, o, t);
    else tandem::fill_normal(p, (device float *)out, t);
    """,
    header=HEADER,
    compile_options=_OPTIONS,
)

_fill_normal_odd = mx.fast.metal_kernel(
    name="tandem_fill_normal_odd",
    input_names=["key", "params"],
    output_names=["out"],
    source="""
    threadgroup uint word0[tandem::THREADS];
    threadgroup uint first0[tandem::GROUPS];
    tandem::Params p = *(const device tandem::Params *)params;
    p.key = uint4(key[0], key[1], key[2], key[3]);
    tandem::fill_normal_odd(p, (device float *)out, thread_position_in_grid.x,
                            thread_index_in_threadgroup, threadgroup_position_in_grid.x, word0, first0);
    """,
    header=HEADER,
    compile_options=_OPTIONS,
)

_fill_choice = mx.fast.metal_kernel(
    name="tandem_fill_choice",
    input_names=["key", "params", "cut", "alias"],
    output_names=["out"],
    source="""
    tandem::Params p = *(const device tandem::Params *)params;
    p.key = uint4(key[0], key[1], key[2], key[3]);
    tandem::fill_choice(p, (device uint *)out, cut, alias, thread_position_in_grid.x);
    """,
    header=HEADER,
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
    header=HEADER,
    compile_options=_OPTIONS,
)


def _threshold(r, w):
    return (2**w - r) % r if r else 0


def fill(key, kind, aligned, n, K, dtype, range_=0, lo=0):
    """`n` elements of `kind` from the aligned bit position, as a flat array of `dtype`. The
    parameters are tandem.metal's Params, packed as tandem-metal's Swift host packs them."""
    if kind in (NORMAL32, NORMAL32_ODD):
        s0 = aligned // 32
        last = s0 + 2 * -(-n // 2) - 1
        g0, g1 = (s0 >> 5) // K, (last >> 5) // K
        b0 = b1 = thresh = 0
    else:
        w = _WIDTH.get(kind, 32)
        b0, b1 = aligned // 8, (aligned + w * n) // 8
        g0, g1 = (b0 >> 7) // K, ((b1 - 1) >> 7) // K
        s0, thresh = 0, _threshold(range_, w)
    # Words 0 to 3 hold the key, which the kernel reads from its own input.
    params = np.array([0, 0, g0, b0, b1, range_, lo % 2**64, thresh, n, s0, 0, K], np.uint64).view(np.uint32)
    kernel, template = (_fill_normal_odd, None) if kind == NORMAL32_ODD else (_fill, [("KIND", kind)])
    threads = -(-8 * (g1 - g0 + 1) // _THREADS) * _THREADS
    return kernel(
        inputs=[key, mx.array(params)],
        template=template,
        grid=(threads, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[dtype],
    )[0]


def fill_choice(key, aligned, n, K, capacity, cut, alias):
    """`n` uint32 indices of the alias table (capacity, cut, alias) from the aligned bit position,
    one 64-bit draw each. Params holds m as the range and the capacity as the threshold."""
    b0, b1 = aligned // 8, (aligned + 64 * n) // 8
    g0, g1 = (b0 >> 7) // K, ((b1 - 1) >> 7) // K
    params = np.array([0, 0, g0, b0, b1, cut.size, 0, capacity, n, 0, 0, K], np.uint64).view(np.uint32)
    threads = -(-8 * (g1 - g0 + 1) // _THREADS) * _THREADS
    return _fill_choice(
        inputs=[key, mx.array(params), mx.array(cut), mx.array(alias)],
        grid=(threads, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.uint32],
    )[0]


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
