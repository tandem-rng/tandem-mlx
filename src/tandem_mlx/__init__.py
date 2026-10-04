"""Tandem8x32 for MLX: the specification's stream and the derived draws of its Appendix A,
computed on the GPU by the Metal kernels of tandem.metal.

A key is a (4,) uint32 array. Draw functions read the key's stream from a bit position, with
the specification's alignment, and the `stream*` forms also return the position after the draws.
"""

import math

import mlx.core as mx
import numpy as np

from . import _kernels as _k

__all__ = [
    "key", "split", "fork", "purpose", "stream", "bits", "uniform", "stream_normal", "normal",
    "stream_exponential", "exponential", "stream_randint", "randint", "metal_source",
]

K = 32
"""The default chunk length: the canonical Tandem8x32-K32."""

DOMAIN_SPLIT = 0xBB67AE85
DOMAIN_FORK = 0xD2511F53
DOMAIN_FOLD = 0xCD9E8D57
DOMAIN_SEED = 0xA54FF53A

_INT_WIDTH = {mx.uint8: 8, mx.uint16: 16, mx.uint32: 32, mx.uint64: 64, mx.int8: 8, mx.int16: 16, mx.int32: 32, mx.int64: 64}
_UNSIGNED = {8: mx.uint8, 16: mx.uint16, 32: mx.uint32, 64: mx.uint64}


def metal_source():
    """tandem.metal without its kernels, as a header for `mx.fast.metal_kernel`."""
    return _k.HEADER


def _key(k):
    k = k if isinstance(k, mx.array) else mx.array(np.asarray(k, np.uint32))
    if k.shape != (4,) or k.dtype != mx.uint32:
        raise ValueError(f"a key is a (4,) uint32 array, got {k.shape} {k.dtype}")
    return k


def _chunk(K):
    if not 1 <= K <= 65536 or K & (K - 1):
        raise ValueError(f"chunk length must be a power of two from 1 to 65536, got {K}")
    return K


def _rows(counter, aux, half):
    """uint32 rows (counter lo, counter hi, aux, half) for the derive kernel."""
    counter = np.asarray(counter, np.uint64)
    aux, half = np.broadcast_to(aux, counter.shape), np.broadcast_to(half, counter.shape)
    rows = np.stack([counter & 0xFFFFFFFF, counter >> np.uint64(32), aux, half], -1).astype(np.uint32)
    return rows.reshape(-1, 4), counter.shape


def _derive(k, counter, aux, half, domain):
    rows, shape = _rows(counter, aux, half)
    if rows.shape[0] == 0:
        return mx.zeros((*shape, 4), mx.uint32)
    return _k.derive(k, rows, domain).reshape(*shape, 4)


def _unsigned_index(x, what):
    x = np.asarray(x.tolist() if isinstance(x, mx.array) else x, dtype=object)
    if x.size and (x.min() < 0 or x.max() >= 2**64):
        raise ValueError(f"{what} must lie in [0, 2**64)")
    return x.astype(np.uint64)


def key(seed):
    """The key of an integer seed in [0, 2**128) through the specification's seed whitening, so
    `key(42)` is the generator of C `tandem_seed(42, 0, 32)` and Julia `Tandem8x32(42)`."""
    seed = int(seed)
    if not 0 <= seed < 2**128:
        raise ValueError("seed must lie in [0, 2**128)")
    words = mx.array(np.array([(seed >> (32 * i)) & 0xFFFFFFFF for i in range(4)], np.uint32))
    return _derive(words, 0, 0, 0, DOMAIN_SEED)


def split(k, index):
    """The spec's split child `index` of a key, for an unsigned 64-bit index or an array of
    them: `half(index & 1)` of F(key, index >> 1, DOMAIN_SPLIT, 0). Children of an array have
    the shape of the index plus (4,)."""
    i = _unsigned_index(index, "index")
    return _derive(_key(k), i >> np.uint64(1), 0, i & np.uint64(1), DOMAIN_SPLIT)


def purpose(k, u):
    """The spec's purpose child of a key for an unsigned 64-bit identifier `u`, or an array."""
    return _derive(_key(k), _unsigned_index(u, "purpose"), 0, 0, DOMAIN_FOLD)


def fork(k, position, n):
    """The `n` fork children of a key at bit `position` as an (n, 4) array, and the parent's
    new position, the start of the block after the one that holds `position`."""
    if not 0 <= n <= 2**33:
        raise ValueError("a fork batch holds 0 to 2**33 children")
    b = _position(position) >> 7
    i = np.arange(n, dtype=np.uint64)
    kids = _derive(_key(k), np.full(n, b, np.uint64), (i >> np.uint64(1)) & np.uint64(0xFFFFFFFF), i & np.uint64(1), DOMAIN_FORK)
    return kids, (b + 1) << 7


def _position(position):
    position = int(position)
    if not 0 <= position < 2**64:
        raise ValueError("a position lies in [0, 2**64)")
    return position


def _start(position, w, n):
    """The aligned start of `n` draws of width `w`, checked against the end of the stream."""
    aligned = (_position(position) + w - 1) & ~(w - 1)
    if aligned + w * n >= 2**64:
        raise ValueError("the draws run past bit 2**64 of the stream")
    return aligned


def _shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _words(k, aligned, n, w, K):
    """`n` unsigned draws of width w <= 32 from the aligned position, read out of 32-bit words."""
    if w == 32:
        return _k.fill(k, _k.BITS32, aligned, n, K, mx.uint32)
    start = aligned & ~31
    skip = (aligned - start) // w
    words = _k.fill(k, _k.BITS32, start, -(-(skip + n) * w // 32), K, mx.uint32)
    # Little-endian words hold the narrower draws in stream order.
    return words.view(_UNSIGNED[w])[skip : skip + n]


def _f64(bits):
    # Metal has no double type. The mapping (raw >> 11) * 2^-53 is exact, so the CPU applies it.
    return mx.multiply((bits >> 11).astype(mx.float64, stream=mx.cpu), 2.0**-53, stream=mx.cpu)


def stream(k, position, n, dtype=mx.uint32, chunk_length=K):
    """`n` draws of `dtype` from stream bit `position` and the position after them, with the
    spec's alignment and mappings. Integer types are the unsigned draws, reinterpreted for the
    signed ones. float16, float32 and float64 use the spec's exact mappings. float64 is
    computed on the CPU stream, as Metal has no double type."""
    k, K = _key(k), _chunk(chunk_length)
    if dtype in _INT_WIDTH:
        w = _INT_WIDTH[dtype]
        aligned = _start(position, w, n)
        if n == 0:
            return mx.zeros(0, dtype), aligned
        raw = _k.fill(k, _k.BITS64, aligned, n, K, mx.uint64) if w == 64 else _words(k, aligned, n, w, K)
        return (raw if raw.dtype == dtype else raw.view(dtype)), aligned + w * n
    w = {mx.float16: 16, mx.float32: 32, mx.float64: 64}.get(dtype)
    if w is None:
        raise TypeError(f"stream does not support dtype {dtype}")
    aligned = _start(position, w, n)
    if n == 0:
        out = mx.zeros(0, dtype, stream=mx.cpu if w == 64 else None)
    elif w == 32:
        out = _k.fill(k, _k.UNIFORM32, aligned, n, K, mx.float32)
    elif w == 16:
        # An 11-bit integer and a power-of-two scaling: both steps are exact in float16.
        out = (_words(k, aligned, n, 16, K) >> 5).astype(mx.float16) * mx.array(2.0**-11, mx.float16)
    else:
        out = _f64(_k.fill(k, _k.BITS64, aligned, n, K, mx.uint64))
    return out, aligned + w * n


def bits(k, shape=(), dtype=mx.uint32, position=0, chunk_length=K):
    """The unsigned (or reinterpreted signed) draws of `dtype` from bit `position`, in `shape`."""
    if dtype not in _INT_WIDTH:
        raise TypeError(f"bits needs an integer dtype, got {dtype}")
    shape = _shape(shape)
    return stream(k, position, math.prod(shape), dtype, chunk_length)[0].reshape(shape)


def uniform(k, shape=(), dtype=mx.float32, position=0, *, low=0.0, high=1.0, chunk_length=K):
    """Uniforms on [low, high) from bit `position`, by the spec's mappings for float16, float32
    and float64. On [0, 1) they are the spec's draws. Returns the array only: `stream` also
    returns the next position. float64 lives on the CPU stream."""
    if dtype not in (mx.float16, mx.float32, mx.float64):
        raise TypeError(f"uniform needs float16, float32 or float64, got {dtype}")
    shape, s = _shape(shape), mx.cpu if dtype == mx.float64 else None
    x = mx.reshape(stream(k, position, math.prod(shape), dtype, chunk_length)[0], shape, stream=s)
    if (low, high) == (0, 1):
        return x
    return mx.add(mx.multiply(x, high - low, stream=s), low, stream=s)


def _f32_only(dtype, what):
    if dtype == mx.float64:
        raise TypeError(f"float64 {what} need double precision, which Metal lacks. Use tandem_numpy (tandem-rng) on the host.")
    if dtype != mx.float32:
        raise TypeError(f"{what} need float32, got {dtype}")


def stream_normal(k, position, n, dtype=mx.float32, chunk_length=K):
    """`n` standard normals by Box-Muller from bit `position`, as Appendix A defines them, and
    the position after them. Pair j is elements 2j and 2j + 1, `(r cos 2 pi b, r sin 2 pi b)`
    with `r = sqrt(-2 log(1 - a))`, from uniform draws 2j (a) and 2j + 1 (b). The fill consumes
    2 ceil(n / 2) draws and an odd n drops the last sin half. n = 0 leaves the position as it is.
    float32 only, bit for bit equal to tandem-c."""
    _f32_only(dtype, "normals")
    k, K = _key(k), _chunk(chunk_length)
    if n == 0:
        return mx.zeros(0, mx.float32), _position(position)
    draws = 2 * -(-n // 2)
    aligned = _start(position, 32, draws)
    kind = _k.NORMAL32 if (aligned // 32) % 2 == 0 else _k.NORMAL32_ODD
    return _k.fill(k, kind, aligned, n, K, mx.float32), aligned + 32 * draws


def normal(k, shape=(), dtype=mx.float32, position=0, *, loc=0.0, scale=1.0, chunk_length=K):
    """Normals of `shape` from bit `position`, `loc + scale * z` for the `stream_normal` values."""
    shape = _shape(shape)
    z = stream_normal(k, position, math.prod(shape), dtype, chunk_length)[0].reshape(shape)
    return z if (loc, scale) == (0, 1) else z * scale + loc


def stream_exponential(k, position, n, dtype=mx.float32, chunk_length=K):
    """`n` standard exponentials `-log(1 - u)` from the uniform draws at bit `position`, one draw
    each, as Appendix A defines them, and the position after them. n = 0 leaves the position as
    it is. float32 only, bit for bit equal to tandem-c."""
    _f32_only(dtype, "exponentials")
    k, K = _key(k), _chunk(chunk_length)
    if n == 0:
        return mx.zeros(0, mx.float32), _position(position)
    aligned = _start(position, 32, n)
    return _k.fill(k, _k.EXPONENTIAL32, aligned, n, K, mx.float32), aligned + 32 * n


def exponential(k, shape=(), dtype=mx.float32, position=0, chunk_length=K):
    """Standard exponentials of `shape` from bit `position`, as `stream_exponential` gives them."""
    shape = _shape(shape)
    return stream_exponential(k, position, math.prod(shape), dtype, chunk_length)[0].reshape(shape)


def stream_randint(k, position, n, low, high, dtype=mx.int32, width=None, chunk_length=K):
    """`n` integers uniform on [low, high) by Lemire's method, as Appendix A defines them, and the
    position after them. Element i uses draw i, and a rejected draw retries on the fallback
    stream split(g) of purpose(0x424c573332), or 0x424c573634 for 64-bit draws, with g the draw's
    index in the key's stream. The draw width follows the range, 32 bits up to 2**32 and 64
    above, so the dtype does not change the values. `width` names it instead. `high <= low`
    gives `low`. n = 0 leaves the position as it is."""
    if dtype not in _INT_WIDTH:
        raise TypeError(f"randint needs an integer dtype, got {dtype}")
    k, K = _key(k), _chunk(chunk_length)
    bits_ = _INT_WIDTH[dtype]
    signed = dtype in (mx.int8, mx.int16, mx.int32, mx.int64)
    low, high = int(low), int(high)
    lo_min = -(2 ** (bits_ - 1)) if signed else 0
    if not lo_min <= low <= lo_min + 2**bits_ - 1:
        raise ValueError(f"low = {low} does not fit {dtype}")
    r = max(0, min(high, lo_min + 2**bits_) - low)
    if width not in (None, 32, 64):
        raise ValueError(f"width must be 32 or 64, got {width}")
    w = width or (32 if r <= 2**32 else 64)
    if r > 2**w:
        raise ValueError(f"a range of {r} needs 64-bit draws")
    if n == 0:
        return mx.zeros(0, dtype), _position(position)
    aligned = _start(position, w, n)
    wide = bits_ == 64 or w == 64
    out = (mx.int64 if signed else mx.uint64) if wide else (mx.int32 if signed else mx.uint32)
    if r == 2**32 and w == 32:
        # Lemire on the full 32-bit range accepts every draw as it is.
        raw = _k.fill(k, _k.BITS32, aligned, n, K, mx.uint32)
        u = np.uint64 if wide else np.uint32
        low_bits = mx.array(np.array(low % 2 ** (8 * np.dtype(u).itemsize), u))
        val = (raw.astype(low_bits.dtype) + low_bits).view(out)
    else:
        kind = _k.BELOW64 if w == 64 else _k.BELOW32_64 if wide else _k.BELOW32
        val = _k.fill(k, kind, aligned, n, K, out, r, low)
    return (val if out == dtype else val.astype(dtype)), aligned + w * n


def randint(k, shape, low, high, dtype=mx.int32, position=0, width=None, chunk_length=K):
    """Integers of `shape` uniform on [low, high) from bit `position`, as `stream_randint` gives them."""
    shape = _shape(shape)
    return stream_randint(k, position, math.prod(shape), low, high, dtype, width, chunk_length)[0].reshape(shape)
