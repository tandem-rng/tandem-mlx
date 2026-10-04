"""Spec vectors, tandem-c stream dumps, and the derived-draw fixtures of tandem-c and tandem-cuda."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import tandem_mlx as tm
from tandem_mlx import _kernels

HERE = Path(__file__).parent
V = json.loads((HERE / "vectors.json").read_text())
D = json.loads((HERE / "cross_derived.json").read_text())
KEY = mx.array(np.array([int(w, 16) for w in V["key"]], np.uint32))
KEY1234 = mx.array(np.array([1, 2, 3, 4], np.uint32))
KEY42 = mx.array(np.array([0x421D21EB, 0x32D31777, 0x62E7564B, 0xDF2BDF82], np.uint32))


def words(ws):
    return np.array([int(w, 16) for w in ws], np.uint32)


def dump(name, dtype):
    return np.fromfile(HERE / "data" / name, dtype=dtype)


def bits_of(x):
    x = np.array(x)
    return x.view(np.dtype(f"uint{8 * x.itemsize}"))


# T and F of the shader on arbitrary states, for the spec's vectors.
_state = mx.fast.metal_kernel(
    name="test_state",
    input_names=["s"],
    output_names=["out"],
    source="""
    tandem::State x = {uint4(s[0], s[1], s[2], s[3]), uint4(s[4], s[5], s[6], s[7])};
    x = F_ROUNDS ? tandem::F(x) : tandem::T(x);
    for (uint i = 0; i < 4; i++) out[i] = x.o[i], out[4 + i] = x.h[i];
    """,
    header=tm.metal_source(),
)


def run_state(o, h, f):
    s = mx.array(np.concatenate([o, h]))
    out = _state(inputs=[s], template=[("F_ROUNDS", f)], grid=(1, 1, 1), threadgroup=(1, 1, 1),
                 output_shapes=[(8,)], output_dtypes=[mx.uint32])[0]
    out = np.array(out)
    return out[:4], out[4:]


# A host reference of T and F for blocks far out in the stream.
M32 = 0xFFFFFFFF


def ref_T(o, h):
    rotl = lambda x, r: ((x << r) | (x >> (32 - r))) & M32
    p0, p1 = o[0] * (h[0] | 1), o[2] * (h[1] | 1)
    lo0, hi0, lo1, hi1 = p0 & M32, p0 >> 32, p1 & M32, p1 >> 32
    n = [o[1] ^ hi1 ^ lo1, rotl(lo1, 16) ^ h[2], o[3] ^ hi0 ^ lo0, rotl(lo0, 16) ^ h[3]]
    h = list(h)
    h[0] ^= rotl(h[1], 7)
    h[1] ^= rotl(h[2], 13)
    h[2] ^= rotl(h[3], 22)
    h[3] ^= rotl(h[0], 3)
    h[0] = ((h[0] + 0x9E3779B9) & M32) ^ n[0]
    return n, h


def ref_block(key, c, j):
    o, h = [c & M32, c >> 32, 0x9E3779B9, 0x94D049BB], list(key)
    for rc in (0xD17CC1B7, 0xA7220A94, 0xFE13ABE8, 0xFA9A6EE0, 0xEDB14ACC, 0x9E21C820, 0xFF28B1D5, 0xEF5DE2B0):
        o, h = ref_T(o, h)
        o[0] ^= rc
        o, h = h, o
    for _ in range(j + 1):
        o, h = ref_T(o, h)
    return o


def test_T():
    for t in V["T"]:
        o, h = run_state(words(t["o"]), words(t["h"]), False)
        assert np.array_equal(o, words(t["o_out"])) and np.array_equal(h, words(t["h_out"]))


def test_F():
    for f in V["F"]:
        o, h = run_state(np.array([f["counter"], 0, 0x9E3779B9, 0x94D049BB], np.uint32), np.array(KEY), True)
        assert np.array_equal(o, words(f["o"])) and np.array_equal(h, words(f["h"]))


def test_stream_words():
    got, pos = tm.stream(KEY, 0, 64, mx.uint32)
    assert pos == 64 * 32
    for s in V["stream_words"]:
        i = s["first_word"]
        assert np.array_equal(np.array(got[i : i + 4]), words(s["words"]))


def test_draws_from_position_0():
    d = V["draws_from_position_0"]
    # float64 arrays live on the CPU stream: the default GPU stream cannot index them.
    f64 = np.array(tm.stream(KEY, 0, 17, mx.float64)[0])
    f32, _ = tm.stream(KEY, 0, 3, mx.float32)
    for i, x in d["Float64"].items():
        assert f64[int(i)] == x
    for i, x in d["Float32"].items():
        assert f32[int(i)].item() == np.float32(x)
    u8 = np.array(tm.stream(KEY, 0, 17, mx.uint8)[0])
    for i, b in d["Bool"].items():
        assert (u8[int(i) // 8] >> (int(i) % 8)) & 1 == b


def test_derived_keys():
    k = V["derived_keys"]
    kids = np.array(tm.split(KEY, [0, 1]))
    assert np.array_equal(kids[0], words(k["split_child_0"])) and np.array_equal(kids[1], words(k["split_child_1"]))
    assert np.array_equal(np.array(tm.split(KEY, 1)), words(k["split_child_1"]))
    assert np.array_equal(np.array(tm.purpose(KEY, 7)), words(k["purpose_7"]))
    forks, pos = tm.fork(KEY, 0, 3)
    assert forks.shape == (3, 4) and np.array_equal(np.array(forks[0]), words(k["fork_child_0_at_block_0"]))
    assert pos == 128


def test_seed_whitening():
    s = V["seed_whitening"]
    k = tm.key(s["seed"])
    assert np.array_equal(np.array(k), words(s["key"]))
    f64 = np.array(tm.stream(k, 0, 17, mx.float64)[0])
    for i, x in s["Float64"].items():
        assert f64[int(i)] == x
    assert tm.stream(k, 0, 1, mx.uint32)[0].item() == int(s["UInt32"]["0"], 16)


@pytest.mark.parametrize(
    "name, key, K, dtype",
    [
        ("k1234_K32_u32.bin", KEY1234, 32, mx.uint32),
        ("k1234_K32_u64.bin", KEY1234, 32, mx.uint64),
        ("k1234_K8_u32.bin", KEY1234, 8, mx.uint32),
        ("seed42_K32_f64.bin", 42, 32, mx.float64),
        ("seed42_K32_f32.bin", 42, 32, mx.float32),
        ("seed42_K32_u8.bin", 42, 32, mx.uint8),
        ("seed42_K32_f16bits.bin", 42, 32, mx.float16),
    ],
)
def test_dumps(name, key, K, dtype):
    want = dump(name, np.uint16 if dtype == mx.float16 else np.dtype(str(dtype).split(".")[-1]))
    k = tm.key(key) if isinstance(key, int) else key
    w = 8 * want.itemsize
    got, pos = tm.stream(k, 0, len(want), dtype, chunk_length=K)
    assert np.array_equal(bits_of(got), bits_of(want)) and pos == len(want) * w
    # Positioned reads, from mid-row, mid-block and mid-word starts and unaligned bits.
    for start in (1, 7, 33, 100, 257):
        part, end = tm.stream(k, start * w - (w > 8), 300, dtype, chunk_length=K)
        assert np.array_equal(bits_of(part), bits_of(want[start : start + 300])) and end == (start + 300) * w


def test_alignment_and_empty_stream():
    k = tm.key(42)
    u8, p = tm.stream(k, 0, 1, mx.uint8)
    assert p == 8
    u64, p = tm.stream(k, p, 1, mx.uint64)
    assert p == 128 and u64.item() == tm.stream(k, 0, 2, mx.uint64)[0][1].item()
    # An empty fill returns the position aligned to the width.
    for dtype, want in ((mx.uint32, 64), (mx.float32, 64), (mx.float64, 64), (mx.int16, 48)):
        x, p = tm.stream(k, 33, 0, dtype)
        assert x.shape == (0,) and p == want


def test_signed_ints_reinterpret():
    k = tm.key(42)
    for signed, unsigned in ((mx.int8, mx.uint8), (mx.int16, mx.uint16), (mx.int32, mx.uint32), (mx.int64, mx.uint64)):
        s, u = tm.bits(k, (5, 7), signed, 99), tm.bits(k, (5, 7), unsigned, 99)
        assert s.dtype == signed and s.shape == (5, 7)
        assert np.array_equal(np.array(s).view(np.array(u).dtype), np.array(u))


def test_blocks_far_out_in_the_stream():
    # Group indices above 2^32 need the 64-bit chunk counter, and the variants place blocks differently.
    for K in (1, 8, 32, 65536):
        for row in (2**50 + 3, 2**54 - 1):
            for lane in (0, 5):
                got = np.array(tm.stream(KEY1234, (row * 8 + lane) * 128, 4, mx.uint32, chunk_length=K)[0])
                assert list(got) == ref_block([1, 2, 3, 4], (row // K) * 8 + lane, row % K), (K, row, lane)


def test_chunk_length_variants():
    want = dump("k1234_K8_u32.bin", np.uint32)
    assert np.array_equal(np.array(tm.bits(KEY1234, len(want), chunk_length=8)), want)
    assert not np.array_equal(np.array(tm.bits(KEY1234, len(want))), want)
    for bad in (0, 3, 131072, -8):
        with pytest.raises(ValueError, match="power of two"):
            tm.bits(KEY1234, 4, chunk_length=bad)


def test_cross_port_fixture_from_c_reference():
    # tests/cross_port.json comes from tandem-jax, written by its tools/gen_split_fixture.c over tandem-c.
    X = json.loads((HERE / "cross_port.json").read_text())
    k = tm.key(X["seed"])
    assert np.array_equal(np.array(k), words(X["key"]))
    split = np.array(tm.split(k, np.array([int(i) for i in X["split"]], np.uint64)))
    for got, want in zip(split, X["split"].values()):
        assert np.array_equal(got, words(want))
    for p, want in X["fork"].items():
        kids, new = tm.fork(k, int(p), 3)
        assert new == want["new_position"]
        assert np.array_equal(np.array(kids), np.array([words(w) for w in want["children"]]))
    subs = np.array(tm.purpose(k, [int(u) for u in X["sub"]]))
    for got, want in zip(subs, X["sub"].values()):
        assert np.array_equal(got, words(want))
    assert tm.split(k, np.zeros((0,), np.uint64)).shape == (0, 4) and tm.fork(k, 300, 0)[0].shape == (0, 4)


def test_argument_checks():
    with pytest.raises(ValueError):
        tm.key(2**128)
    with pytest.raises(ValueError):
        tm.split(KEY, -1)
    with pytest.raises(ValueError):
        tm.split(KEY, 2**64)
    with pytest.raises(ValueError):
        tm.bits(mx.zeros((3,), mx.uint32), 4)
    with pytest.raises(ValueError):
        tm.stream(KEY, 2**64 - 64, 1, mx.uint64)
    with pytest.raises(ValueError):
        tm.fork(KEY, 0, 2**33 + 1)
    with pytest.raises(TypeError):
        tm.stream(KEY, 0, 4, mx.bfloat16)
    with pytest.raises(TypeError):
        tm.uniform(KEY, 4, mx.int32)


def test_uniform():
    k = tm.key(42)
    for dtype, name, np_dtype in ((mx.float32, "seed42_K32_f32.bin", np.float32), (mx.float64, "seed42_K32_f64.bin", np.float64)):
        want = dump(name, np_dtype)
        x = tm.uniform(k, (10, 100), dtype, position=8 * want.itemsize * 3)
        assert x.dtype == dtype and np.array_equal(np.array(x).reshape(-1), want[3:1003])
    y = np.array(tm.uniform(k, 1000, low=-2.0, high=3.0))
    assert np.array_equal(y, (dump("seed42_K32_f32.bin", np.float32)[:1000] * 5 - 2).astype(np.float32))
    assert (y >= -2).all() and (y < 3).all()
    assert tm.uniform(k, 7, mx.float64, low=1, high=2).dtype == mx.float64


# ---- Appendix A ------------------------------------------------------------------------------


def test_normal_pairs_match_c_reference():
    z, pos = tm.stream_normal(tm.key(42), 1, len(D["pairs32"]))
    assert z.dtype == mx.float32 and pos == D["pairs32_end_pos"]
    assert np.array_equal(bits_of(z), bits_of(np.array(D["pairs32"], np.float32)))


def test_normal_fills_match_cuda_fixtures():
    # Starts 0, 64 and 1000 run the one-pass kernel, 32 and 96 the pair pass from an odd draw.
    for c in D["device_normal32"]:
        z, pos = tm.stream_normal(KEY42, c["pos"], c["n"])
        assert np.array_equal(bits_of(z), bits_of(np.array(c["out"], np.float32))), c["pos"]
        assert pos == (c["pos"] + 31) // 32 * 32 + 32 * 2 * ((c["n"] + 1) // 2)
        assert np.array_equal(np.array(tm.normal(KEY42, c["n"], position=c["pos"])), np.array(z))


def test_normal_edge_cases():
    k = tm.key(42)
    z, pos = tm.stream_normal(k, 5, 0)
    assert z.shape == (0,) and pos == 5
    for start in (0, 32, 64, 96):
        odd, p_odd = tm.stream_normal(k, start, 5)
        even, p_even = tm.stream_normal(k, start, 6)
        assert np.array_equal(np.array(odd), np.array(even[:5])) and p_odd == p_even == start + 6 * 32
    big = np.array(tm.normal(k, (1000, 1000)))
    assert abs(big.mean()) < 0.005 and abs(big.std() - 1) < 0.005
    scaled = np.array(tm.normal(k, 100, loc=3.0, scale=2.0))
    assert np.array_equal(scaled, np.array(tm.normal(k, 100)) * np.float32(2) + np.float32(3))
    with pytest.raises(TypeError, match="Metal"):
        tm.normal(k, 4, mx.float64)
    with pytest.raises(TypeError):
        tm.normal(k, 4, mx.float16)


_pairs = mx.fast.metal_kernel(
    name="test_pairs",
    input_names=["u"],
    output_names=["out"],
    source="""
    uint j = thread_position_in_grid.x;
    float2 z = tandem::normal_pair_f32(u[2 * j], u[2 * j + 1]);
    out[2 * j] = z.x, out[2 * j + 1] = z.y;
    """,
    header=tm.metal_source(),
)


@pytest.mark.parametrize("K", [1, 8, 32])
def test_normal_fills_equal_pairs_of_the_uniforms(K):
    # One pass from an even draw, and the shuffled one from an odd draw, against Box-Muller
    # applied to the uniform fill. Fills run over many groups, end in every lane, and start at
    # every word of a block and far out in the stream.
    k = tm.key(11)
    for start in (0, 32, 64, 96, 2**45 + 32, 2**45 + 1000):
        for n in (1, 2, 3, 61, 62, 4097, 30000):
            draws = 2 * -(-n // 2)
            u, _ = tm.stream(k, start, draws, mx.float32, K)
            want = _pairs(inputs=[u], grid=(draws // 2, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(draws,)], output_dtypes=[mx.float32])[0]
            got, pos = tm.stream_normal(k, start, n, chunk_length=K)
            assert np.array_equal(bits_of(got), bits_of(want[:n])), (start, n)
            assert pos == (start + 31) // 32 * 32 + 32 * draws


def test_exponentials_match_c_and_cuda():
    for c in D["exponential32"]:
        e, pos = tm.stream_exponential(tm.key(42), c["start"], len(c["out"]))
        assert np.array_equal(bits_of(e), bits_of(np.array(c["out"], np.float32))) and pos == c["end_pos"], c["start"]
    for c in D["device_exponential32"]:
        e = tm.exponential(KEY42, c["n"], position=c["pos"])
        assert np.array_equal(bits_of(e), bits_of(np.array(c["out"], np.float32))), c["pos"]
    k = tm.key(42)
    e, pos = tm.stream_exponential(k, 7, 0)
    assert e.shape == (0,) and pos == 7
    big = np.array(tm.exponential(k, 10**6))
    assert (big >= 0).all() and abs(big.mean() - 1) < 0.005
    with pytest.raises(TypeError, match="Metal"):
        tm.exponential(k, 4, mx.float64)


@pytest.mark.parametrize("name, dtype, w", [("fill_below32", mx.uint32, 32), ("fill_below64", mx.uint64, 64)])
def test_randint_matches_c_fills(name, dtype, w):
    k = tm.key(42)
    for c in D[name]:
        got, pos = tm.stream_randint(k, c["start"], len(c["out"]), 0, c["range"], dtype, w)
        assert np.array_equal(np.array(got), np.array(c["out"], np.dtype(f"uint{w}"))), (c["start"], c["range"])
        assert pos == c["end_pos"]
    assert {c["start"] for c in D[name]} >= {0, 1, 12345}


@pytest.mark.parametrize("name, w", [("scalar_below32", 32), ("scalar_below64", 64)])
def test_scalar_bounded_fixtures_match_sequential_lemire(name, w):
    # A scalar draw rejects by taking the next draw of the main stream, which a fill does not do,
    # so the fixture is checked against Lemire's loop over the plain draws.
    draws = [int(x) for x in np.array(tm.stream(tm.key(42), 1, 600, tm._UNSIGNED[w])[0])]
    for c in D[name]:
        r, out, used = c["range"], [], 0
        t = ((1 << w) - r) % r
        while len(out) < len(c["out"]):
            m = draws[used] * r
            used += 1
            if m % (1 << w) >= t:
                out.append(m >> w)
        assert out == c["out"] and w + w * used == c["end_pos"], r


@pytest.mark.parametrize(
    "name, dtype, w",
    [("device_below32", mx.uint32, 32), ("device_below64", mx.uint64, 64), ("device_below32_at", mx.uint32, 32), ("device_below64_at", mx.uint64, 64)],
)
def test_randint_matches_cuda_fills_with_rejections(name, dtype, w):
    assert max(c["rejected"] for c in D[name]) > 30
    for c in D[name]:
        got = tm.randint(KEY42, 64, 0, c["range"], dtype, c.get("start", 0), w)
        assert np.array_equal(np.array(got), np.array(c["out"], np.dtype(f"uint{w}"))), (c.get("start"), c["range"])
    if name.endswith("_at"):
        assert len(D[name]) >= 12 and {c["start"] for c in D[name]} == {1, 12345}


def test_randint_cut_equals_whole_at_rejections():
    # Ranges just above half the draw space reject about half the draws.
    k = tm.key(3)
    for dtype, r in ((mx.uint32, 2**31 + 1), (mx.uint64, 2**63 + 1), (mx.int64, 2**31 + 1)):
        for start in (0, 1, 12345, 100000):
            whole, end = tm.stream_randint(k, start, 300, 0, r, dtype)
            for cut in (1, 7, 100, 299):
                a, p = tm.stream_randint(k, start, cut, 0, r, dtype)
                b, q = tm.stream_randint(k, p, 300 - cut, 0, r, dtype)
                assert np.array_equal(np.concatenate([np.array(a), np.array(b)]), np.array(whole)) and q == end
    # Fills that share a draw share its fallback, rejected elements included.
    x = np.array(tm.randint(k, 300, 0, 2**31 + 1, mx.uint32, 0))
    y = np.array(tm.randint(k, 299, 0, 2**31 + 1, mx.uint32, 32))
    plain = (np.array(tm.bits(k, 300)).astype(np.uint64) * (2**31 + 1)) >> 32
    assert (plain != x).sum() > 50 and np.array_equal(x[1:], y)


def test_randint_width_dtype_and_bounds():
    k = tm.key(42)
    a = np.array(tm.randint(k, 1000, -50, 50, mx.int32, 64))
    b = np.array(tm.randint(k, 1000, -50, 50, mx.int64, 64))
    c = np.array(tm.randint(k, 1000, -50, 50, mx.int8, 64))
    assert np.array_equal(a, b) and np.array_equal(a, c) and a.min() == -50 and a.max() == 49
    assert tm.stream_randint(k, 64, 10, 0, 2**32, mx.int64)[1] == 64 + 320
    assert tm.stream_randint(k, 64, 10, 0, 2**32 + 1, mx.int64)[1] == 64 + 640
    # A range of exactly 2^32 on 32-bit draws gives the draws themselves.
    raw = np.array(tm.bits(k, 100, mx.uint32, 64))
    assert np.array_equal(np.array(tm.randint(k, 100, 0, 2**32, mx.uint32, 64)), raw)
    assert np.array_equal(np.array(tm.randint(k, 100, -(2**31), 2**31, mx.int64, 64)), raw.astype(np.int64) - 2**31)
    assert np.array_equal(np.array(tm.randint(k, 50, 9, 3)), np.full(50, 9))
    x, p = tm.stream_randint(k, 7, 0, 0, 10)
    assert x.shape == (0,) and p == 7
    # A width of 64 on a small range gives 64-bit Lemire values, which differ from 32-bit ones.
    w64 = np.array(tm.randint(k, 100, 0, 1000, mx.int64, 0, 64))
    assert np.array_equal(w64, ((np.array(tm.bits(k, 100, mx.uint64)).astype(object) * 1000) >> 64).astype(np.int64))
    assert np.array_equal(np.array(tm.randint(k, 100, 0, 1000, mx.int32, 0, 64)), w64)
    big = np.array(tm.randint(k, (500, 400), -(2**62), 2**62, mx.int64))
    assert big.shape == (500, 400) and big.min() < -(2**61) and big.max() > 2**61
    with pytest.raises(ValueError):
        tm.randint(k, 4, -1, 5, mx.uint32)
    with pytest.raises(ValueError):
        tm.randint(k, 4, 0, 2**33, mx.int64, width=32)
    with pytest.raises(TypeError):
        tm.randint(k, 4, 0, 5, mx.float32)


def test_windows_equal_the_whole_fill(monkeypatch):
    # Fills over the window size run as several launches at offsets and must equal one launch.
    k, n = tm.key(9), 5000
    calls = [
        lambda: tm.stream(k, 3, n, mx.uint32)[0],
        lambda: tm.stream(k, 3, n, mx.uint64)[0],
        lambda: tm.stream(k, 3, n, mx.uint8)[0],
        lambda: tm.stream(k, 3, n, mx.float64)[0],
        lambda: tm.uniform(k, n, position=3),
        lambda: tm.normal(k, n, position=3),
        lambda: tm.normal(k, n - 1, position=40),
        lambda: tm.exponential(k, n, position=3),
        lambda: tm.randint(k, n, 0, 2**31 + 1, mx.uint32, 3),
        lambda: tm.randint(k, n, 0, 2**63 + 1, mx.uint64, 3),
        lambda: tm.randint(k, n, -7, 2**31, mx.int64, 3),
    ]
    whole = [np.array(f()) for f in calls]
    monkeypatch.setattr(_kernels, "WINDOW", 998)
    for f, want in zip(calls, whole):
        assert np.array_equal(np.array(f()), want)


def test_matches_tandem_numpy():
    tr = pytest.importorskip("tandem_rng")
    for start in (0, 32, 1000, 2**40 + 96):
        t = tr.Tandem.from_key(np.array(KEY42), start, 32)
        n = 4097
        assert np.array_equal(t.raw(n, np.uint32), np.array(tm.bits(KEY42, n, mx.uint32, start)))
        t.position = start
        assert np.array_equal(t.random(n, np.float32), np.array(tm.uniform(KEY42, n, position=start)))
        t.position = start
        assert np.array_equal(t.random(n), np.array(tm.uniform(KEY42, n, mx.float64, position=start)))
        t.position = start
        assert np.array_equal(t.below(2**31 + 1, n, np.uint32), np.array(tm.randint(KEY42, n, 0, 2**31 + 1, mx.uint32, start)))
        t.position = start
        assert np.array_equal(bits_of(t.normal(n, np.float32)), bits_of(tm.normal(KEY42, n, position=start)))
        assert t.position == tm.stream_normal(KEY42, start, n)[1]
        t.position = start
        assert np.array_equal(bits_of(t.exponential(n, np.float32)), bits_of(tm.exponential(KEY42, n, position=start)))
