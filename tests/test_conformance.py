"""The spec's conformance files, copies of tandem-spec f420545 conformance/*.json that CI checks
byte for byte, and the items of its conformance/CHECKLIST.md at b31af72. Metal has no double type, so the
Float64 cases and the long outputs of tandem-c's dump tools, which hold Float64 draws or need an
FNV-1a pass over 40 MB, are left to the ports with Float64 draws. MLX has no complex draws."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import tandem_mlx as tm

HERE = Path(__file__).parent
FILES = {name: json.loads((HERE / "conformance" / f"{name}.json").read_text()) for name in ("below", "fill_below", "normal", "exponential", "choice", "hashes")}
KINDS = ("fill_below_u32", "fill_below_u64", "fill_normal_f32", "fill_exponential_f32", "fill_choice")
FILLS = [c for name in ("fill_below", "normal", "exponential", "choice") for c in FILES[name]["cases"] if c["kind"] in KINDS]
WIDTH = {"fill_below_u32": 32, "fill_below_u64": 64, "fill_normal_f32": 32, "fill_exponential_f32": 32, "fill_choice": 64}


def named(name, id):
    return next(c for c in FILES[name]["cases"] if c["id"].endswith(" " + id))


def key(c):
    return mx.array(np.array([int(w, 16) for w in c["key"]], np.uint32))


def bits(c):
    """The values of a case as uint32 or uint64 bit patterns."""
    w = 64 if c["kind"] == "fill_below_u64" else 32
    return np.array([int(v, 16) for v in c["values"]], np.dtype(f"uint{w}"))


def as_bits(x):
    x = np.array(x)
    return x.view(np.dtype(f"uint{8 * x.itemsize}"))


def end(c):
    """The end position: the case's own, or the rule for the draws a fill consumes."""
    w, n = WIDTH[c["kind"]], c["n"]
    if c["kind"] == "fill_normal_f32":
        n = 2 * ((n + 1) // 2)
    elif n == 0 and c["kind"] != "fill_choice":
        return c["start"]
    return c.get("end", -(-c["start"] // w) * w + w * n)


def fill(c):
    """The port's fill for a case, as `f(key, position, n) -> (values, end)`."""
    kind = c["kind"]
    if kind.startswith("fill_below"):
        w = WIDTH[kind]
        return lambda k, p, n: tm.stream_randint(k, p, n, 0, int(c["range"], 16), mx.uint32 if w == 32 else mx.uint64, w, c["K"])
    if kind == "fill_choice":
        table = tm.choice_table(np.array([int(v, 16) for v in c["weights"]], np.uint64).view(np.float64))
        return lambda k, p, n: tm.stream_choice(k, p, n, table, c["K"])
    f = tm.stream_normal if "normal" in kind else tm.stream_exponential
    return lambda k, p, n: f(k, p, n, mx.float32, c["K"])


def test_fill_cases_whole_cut_and_one_element_at_a_time():
    # Float32 normal fills are cut at pair boundaries only, as the checklist says.
    assert len(FILLS) == 114 and sum(c["n"] == 0 for c in FILLS) == 5
    for c in FILLS:
        k, f, n = key(c), fill(c), c["n"]
        got, pos = f(k, c["start"], n)
        assert np.array_equal(as_bits(got), bits(c)) and pos == end(c), c["id"]
        cuts = (2, 8, 20, (n - 1) & ~1) if c["kind"] == "fill_normal_f32" else (1, 7, 20, 21, n - 1)
        for cut in sorted({x for x in cuts if 0 < x < n}):
            head, mid = f(k, c["start"], cut)
            tail, pos = f(k, mid, n - cut)
            assert np.array_equal(np.concatenate([as_bits(head), as_bits(tail)]), bits(c)) and pos == end(c), (c["id"], cut)
        if n and c["kind"] != "fill_normal_f32":
            pos, parts = c["start"], []
            for _ in range(n):
                x, pos = f(k, pos, 1)
                parts.append(as_bits(x))
            assert np.array_equal(np.concatenate(parts), bits(c)) and pos == end(c), c["id"]


def test_scalar_bounded_draws_retry_on_the_next_draw():
    # MLX has no scalar bounded draw. Lemire's loop over the port's plain draws, where a
    # rejection takes the next draw, must give the scalar cases.
    for c in FILES["below"]["cases"]:
        w, r = int(c["kind"][-2:]), int(c["range"], 16)
        draws = [int(x) for x in np.array(tm.stream(key(c), c["start"], 4 * c["n"], mx.uint32 if w == 32 else mx.uint64)[0])]
        t, out, used = ((1 << w) - r) % r, [], 0
        while len(out) < c["n"]:
            m = draws[used] * r
            used += 1
            if m % (1 << w) >= t:
                out.append(m >> w)
        assert out == [int(v, 16) for v in c["values"]], c["id"]
        assert -(-c["start"] // w) * w + w * used == c["end"], c["id"]


def test_fallback_is_keyed_by_the_global_draw_index():
    for name, a, b in (("fill_below", "CROSS_BELOW32_AT[4]", "CROSS_BELOW32[4]"), ("fill_below", "CROSS_BELOW64_AT[6]", "CROSS_BELOW64[6]"),
                       ("choice", "CROSS_CHOICE[1]", "CROSS_CHOICE[0]")):
        assert np.array_equal(bits(named(name, a))[:63], bits(named(name, b))[1:]), a
    assert named("fill_below", "CROSS_BELOW32_AT[4]")["rejected"] > 0


def test_width_follows_the_range():
    c32, c64 = named("fill_below", "CROSS_BELOW32[3]"), named("fill_below", "CROSS_BELOW64[3]")
    got, pos = tm.stream_randint(key(c32), 0, 64, 0, 1000, mx.uint64)
    assert np.array_equal(np.array(got), bits(c32).astype(np.uint64)) and pos == 64 * 32
    assert not np.array_equal(np.array(got), bits(c64))
    # Without a width an int64 result takes it from the range, which every case fitting int64 and
    # needing 64 bits, or a 32-bit case, must match.
    for c in FILES["fill_below"]["cases"]:
        r = int(c["range"], 16)
        if (c["kind"] == "fill_below_u32" or r > 2**32) and r < 2**63:
            auto = tm.stream_randint(key(c), c["start"], c["n"], 0, r, mx.int64)[0]
            assert np.array_equal(np.array(auto).astype(np.uint64), bits(c).astype(np.uint64)), c["id"]
    # Range 0: low, and one draw of the width.
    got, pos = tm.stream_randint(key(c32), 0, 1, 5, 5, mx.int64)
    assert int(got[0].item()) == 5 and pos == 32


def test_float32_normals_pair_up():
    pairs, at32 = named("normal", "CROSS_NORMALF"), named("normal", "CROSS_NORMAL32[1]")
    assert np.array_equal(bits(pairs)[:33], bits(at32))
    assert np.array_equal(bits(named("normal", "CROSS_NORMAL32[2]"))[:31], bits(named("normal", "CROSS_NORMAL32[0]"))[2:33])
    # A fill of one is the cos half and consumes both draws of its pair.
    z, pos = tm.stream_normal(key(at32), 32, 1)
    assert as_bits(z)[0] == bits(at32)[0] and pos == 32 + 64


def test_choice_tables():
    for c in FILES["choice"]["cases"]:
        t = tm.choice_table(np.array([int(v, 16) for v in c["weights"]], np.uint64).view(np.float64))
        assert f"{t.capacity:016x}" == c["capacity"], c["id"]
        if "cut" in c:
            assert [f"{int(x):016x}" for x in t.cut] == c["cut"] and [f"{int(x):08x}" for x in t.alias] == c["alias"], c["id"]
    single = named("choice", "CROSS_CHOICE[3]")
    assert len(single["weights"]) == 1 and not bits(single).any()
    for w in ([], [1, -1], [1, np.nan], [1, np.inf], [0.0, -0.0]):
        with pytest.raises(ValueError):
            tm.choice_table(w)
    t = tm.choice_table([3, 0, 1, 7.5, 0.125])
    x = np.array(tm.choice(tm.key(42), (10, 300), t, 33))
    assert x.shape == (10, 300) and np.array_equal(x.reshape(-1), np.array(tm.stream_choice(tm.key(42), 33, 3000, t)[0]))
    assert not (x == 1).any()


STREAM_DTYPES = {"UInt32": mx.uint32, "UInt64": mx.uint64, "Float64": mx.float64, "Float32": mx.float32, "UInt8": mx.uint8, "Float16": mx.float16}


def test_stream_hashes():
    # MLX has no UInt128, Bool, complex or Char draws.
    done = 0
    for s in FILES["hashes"]["streams"]:
        path = HERE / "data" / Path(s["file"]).name
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == s["sha256"]
        if s["type"] in STREAM_DTYPES:
            x = np.array(tm.stream(key(s), s["start"], s["n"], STREAM_DTYPES[s["type"]], s["K"])[0])
            assert hashlib.sha256(x.tobytes()).hexdigest() == s["sha256"], s["file"]
            done += 1
    assert done == 7


def test_block_and_2_63_boundaries():
    k = key(FILES["hashes"]["streams"][0])
    # Reads at any position equal the sequential fill, across blocks, rows and chunks of K = 8.
    whole = np.array(tm.stream(k, 0, 3000, mx.uint32, 8)[0])
    for p in (3, 4, 31, 32, 255, 256, 2047, 2048, 2900):
        assert np.array_equal(np.array(tm.stream(k, 32 * p, 100, mx.uint32, 8)[0]), whole[p : p + 100]), p
    # Positions are arguments of a stateless draw, so any one below 2^64 is a valid successor.
    assert tm.stream(k, 2**63 - 1, 1, mx.uint64)[1] == 2**63 + 64
    assert tm.stream(k, 2**64 - 128, 1, mx.uint64)[1] == 2**64 - 64
    t = tm.choice_table([1.0])
    for f in (lambda: tm.stream(k, 2**64 - 64, 1, mx.uint64), lambda: tm.stream_normal(k, 2**64 - 100, 4),
              lambda: tm.stream_exponential(k, 2**64 - 32, 1), lambda: tm.stream_randint(k, 2**64 - 64, 2, 0, 9),
              lambda: tm.stream_choice(k, 2**64 - 64, 1, t)):
        with pytest.raises(ValueError, match="2\\*\\*64"):
            f()
