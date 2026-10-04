// Tandem8x32 for Metal: the building blocks of https://github.com/tandem-rng/spec, fills of the
// stream, and the bounded, normal and exponential draws of its Appendix A.
// Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
//
// Metal has no double type, so Float64 draws are a host mapping of the u64 fill.
//
// Compile with fast math off (MTLMathMode.safe): the normals and exponentials are bit exact with
// tandem-c only with correctly rounded products, sums, division and square root.
//
// This file is the single shader source of the Metal ports. Everything but the kernels lives in
// namespace tandem and reads its parameters from the thread, so tandem-mlx includes the file as
// the header of `mx.fast.metal_kernel` with TANDEM_NO_KERNELS defined.

#include <metal_stdlib>

namespace tandem {
using namespace metal;

constant uint CLOCK_WEYL = 0x9e3779b9u;
constant uint DOMAIN_STREAM = 0x9e3779b9u;
constant uint DOMAIN_SPLIT = 0xbb67ae85u;
constant uint DOMAIN_FOLD = 0xcd9e8d57u;
constant uint AUX_STREAM = 0x94d049bbu;
constant ulong PURPOSE_BELOW32 = 0x424c573332ul;
constant ulong PURPOSE_BELOW64 = 0x424c573634ul;

struct State {
    uint4 o;
    uint4 h;
};

// The step T: mix, clock, feedback.
static inline State T(State s) {
    uint m0 = s.h.x | 1u, m1 = s.h.y | 1u;
    uint lo0 = s.o.x * m0, hi0 = mulhi(s.o.x, m0);
    uint lo1 = s.o.z * m1, hi1 = mulhi(s.o.z, m1);
    uint4 n = uint4(s.o.y ^ hi1 ^ lo1, rotate(lo1, 16u) ^ s.h.z, s.o.w ^ hi0 ^ lo0,
                    rotate(lo0, 16u) ^ s.h.w);
    uint4 h = s.h;
    h.x ^= rotate(h.y, 7u);
    h.y ^= rotate(h.z, 13u);
    h.z ^= rotate(h.w, 22u);
    h.w ^= rotate(h.x, 3u);
    h.x = (h.x + CLOCK_WEYL) ^ n.x;
    return State{n, h};
}

static inline State seed_round(State s0, uint rc) {
    State s = T(s0);
    return State{s.h, uint4(s.o.x ^ rc, s.o.yzw)};
}

// Written out, since a constant array indexed in a loop may land in memory.
static inline State F(State s) {
    s = seed_round(s, 0xd17cc1b7u);
    s = seed_round(s, 0xa7220a94u);
    s = seed_round(s, 0xfe13abe8u);
    s = seed_round(s, 0xfa9a6ee0u);
    s = seed_round(s, 0xedb14accu);
    s = seed_round(s, 0x9e21c820u);
    s = seed_round(s, 0xff28b1d5u);
    return seed_round(s, 0xef5de2b0u);
}

static inline State F_keyed(uint4 key, ulong counter, uint domain, uint aux) {
    return F(State{uint4(uint(counter), uint(counter >> 32), domain, aux), key});
}

// Block B(c, j): the exposed half of chunk c after j + 1 steps.
static uint4 block(uint4 key, ulong c, uint j) {
    State s = F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM);
    for (uint i = 0; i <= j; i++) s = T(s);
    return s.o;
}

static uint4 split_key(uint4 key, ulong i) {
    State s = F_keyed(key, i >> 1, DOMAIN_SPLIT, 0u);
    return (i & 1) ? s.h : s.o;
}

static uint4 purpose_key(uint4 key, ulong u) { return F_keyed(key, u, DOMAIN_FOLD, 0u).o; }

// ---- Fills ----------------------------------------------------------------------------------
//
// One thread per chunk. A thread walks the K blocks of its chunk and stores each block that
// meets the fill, 16 bytes at a time. The eight lanes of a group are eight consecutive threads
// and their blocks form one 128-byte row, so a SIMD group writes whole rows.

// Threads per threadgroup. The odd normal fill needs exactly this many in every threadgroup.
constant uint THREADS = 256;
constant uint GROUPS = THREADS / 8;

// The 24 words that a host packs, 64-bit values low word first.
struct Params {
    uint4 key;
    ulong g0;      // first group of the dispatch
    ulong b0, b1;  // the fill's stream bytes, [b0, b1)
    ulong range, low, thresh;
    ulong n, s0;   // normals: count and index of the first f32 draw
    ulong out_off; // byte offset of `out` in its buffer, for the alignment of vector stores
    uint K;
    uint pad;
};

// Words of the stream of a key from position 0, for the bounded fallback.
struct Reader {
    uint4 key;
    uint K;
    ulong w;
    uint4 blk;

    uint next() {
        if ((w & 3) == 0) {
            ulong b = w >> 2, row = b >> 3;
            blk = block(key, ((row >> ctz(K)) << 3) + (b & 7), uint(row & (K - 1)));
        }
        return blk[uint(w++ & 3)];
    }
};

// Lemire's multiply and reject, Appendix A. A rejected draw retries on the stream of
// split(g) of purpose(P_w) of the fill's key, where g is the draw's index in the stream, so a
// fill cut at any element equals the whole fill. Rejections are rare, so the retry stays out of
// line. A range of 0 has threshold 0 and gives 0.
static uint below32_retry(thread const Params &P, ulong g) {
    uint range = uint(P.range), t = uint(P.thresh);
    Reader r{split_key(purpose_key(P.key, PURPOSE_BELOW32), g), P.K, 0, uint4(0)};
    uint y;
    do y = r.next();
    while (y * range < t);
    return mulhi(y, range);
}

static ulong below64_retry(thread const Params &P, ulong g) {
    Reader r{split_key(purpose_key(P.key, PURPOSE_BELOW64), g), P.K, 0, uint4(0)};
    ulong y;
    do {
        uint lo = r.next();
        y = ulong(lo) | (ulong(r.next()) << 32);
    } while (y * P.range < P.thresh);
    return mulhi(y, P.range);
}

static inline uint below32(uint x, thread const Params &P, ulong g) {
    uint range = uint(P.range);
    return x * range < uint(P.thresh) ? below32_retry(P, g) : mulhi(x, range);
}

static inline ulong below64(ulong x, thread const Params &P, ulong g) {
    return x * P.range < P.thresh ? below64_retry(P, g) : mulhi(x, P.range);
}

static inline float4 to_f32(uint4 w) { return float4(w >> 8) * 0x1p-24f; }
static inline float to_f32(uint w) { return float(w >> 8) * 0x1p-24f; }

// -2 ln x for x in (0, 1], the logarithm of tandem-c. x = m 2^k with m in [sqrt(1/2), sqrt(2))
// from the bits, then -2 ln x = 2 nk ln 2 - 4 s p with s = (m - 1) / (m + 1) and p a series in
// s^2. ln 2 is split so that nk * ln2_hi is exact.
static inline float neg2_log(float x) {
    uint ix = as_type<uint>(x) + 0x004afb0du;
    float nk = float(127 - int(ix >> 23));
    float m = as_type<float>((ix & 0x007fffffu) + 0x3f3504f3u);
    float s = precise::divide(m - 1.0f, m + 1.0f), z = s * s;
    float p = fma(z, fma(z, fma(z, 0.14275366f, 0.20000061f), 0.33333334f), 1.0f);
    return fma(nk, 2.857213530660374e-06f, fma(nk, 1.38629150390625f, (s * -4.0f) * p));
}

// One Box-Muller step in float, cos half first, the operations of tandem-c's normal loop. The
// angle is cut at the nearest quarter turn q, which is exact, and short series give cos and sin
// on the rest. q then swaps the two and sets their signs.
static float2 normal_pair(float a, float b) {
    float r = precise::sqrt(neg2_log(1.0f - a));
    int q = int(b * 4.0f + 0.5f);
    float f = fma(-float(q), 0.25f, b);
    float th = fma(f, -1.7484555e-7f, f * 6.2831855f), w = th * th;
    float hs = fma(w, fma(w, fma(w, 2.72499e-06f, -0.00019840087f), 0.008333332f), -0.16666667f);
    float hc = fma(w, fma(w, fma(w, 2.4463761e-05f, -0.0013887589f), 0.04166665f), -0.5f);
    float sn = th * fma(w, hs, 1.0f), cs = fma(w, hc, 1.0f);
    uint qu = uint(q), sm = 0u - (qu & 1u);
    uint sb = as_type<uint>(sn), cb = as_type<uint>(cs);
    uint xb = (sb & sm) | (cb & ~sm), yb = (cb & sm) | (sb & ~sm);
    xb ^= ((qu + 1u) << 30) & 0x80000000u;
    yb ^= (qu << 30) & 0x80000000u;
    return r * float2(as_type<float>(xb), as_type<float>(yb));
}

// Output kinds. `make` turns the four words of a stream block into the 16 bytes stored for it,
// `d` is the stream index of the block's first draw.
struct U32 {
    enum { draw = 4 };
    static uint4 make(uint4 w, thread const Params &, ulong) { return w; }
};
struct F32 {
    enum { draw = 4 };
    static uint4 make(uint4 w, thread const Params &, ulong) { return as_type<uint4>(to_f32(w)); }
};
struct Exponential32 {
    enum { draw = 4 };
    static uint4 make(uint4 w, thread const Params &, ulong) {
        float4 u = 1.0f - to_f32(w);
        return as_type<uint4>(0.5f * float4(neg2_log(u.x), neg2_log(u.y), neg2_log(u.z), neg2_log(u.w)));
    }
};
struct Below32 {
    enum { draw = 4 };
    static uint4 make(uint4 w, thread const Params &P, ulong d) {
        uint low = uint(P.low);
        return uint4(below32(w.x, P, d), below32(w.y, P, d + 1), below32(w.z, P, d + 2),
                     below32(w.w, P, d + 3)) + low;
    }
};
struct Below64 {
    enum { draw = 8 };
    static uint4 make(uint4 w, thread const Params &P, ulong d) {
        ulong x = below64(as_type<ulong>(w.xy), P, d) + P.low;
        ulong y = below64(as_type<ulong>(w.zw), P, d + 1) + P.low;
        return uint4(as_type<uint2>(x), as_type<uint2>(y));
    }
};

// Store the part of one block that falls inside the fill. Elements are whole words, so the
// partial path stores words.
template <class E>
static inline void store_block(device uchar *out, thread const Params &P, ulong first, uint4 w) {
    uint4 v = E::make(w, P, first / E::draw);
    if (first >= P.b0 && first + 16 <= P.b1 && ((P.out_off + first - P.b0) & 15) == 0) {
        *(device uint4 *)(out + (first - P.b0)) = v;
        return;
    }
    for (uint k = 0; k < 4; k++) {
        ulong at = first + 4 * k;
        if (at >= P.b0 && at + 4 <= P.b1) *(device uint *)(out + (at - P.b0)) = v[k];
    }
}

// 32-bit bounded draws widened to 64-bit outputs: a block is 32 bytes of output.
static inline void store_block_wide(device uchar *out, thread const Params &P, ulong first, uint4 w) {
    ulong d = first / 4;
    uint2 v[4];
    for (uint k = 0; k < 4; k++) v[k] = as_type<uint2>(ulong(below32(w[k], P, d + k)) + P.low);
    if (first >= P.b0 && first + 16 <= P.b1 && ((P.out_off + 2 * (first - P.b0)) & 15) == 0) {
        device uint4 *dst = (device uint4 *)(out + 2 * (first - P.b0));
        dst[0] = uint4(v[0], v[1]);
        dst[1] = uint4(v[2], v[3]);
        return;
    }
    for (uint k = 0; k < 4; k++) {
        ulong at = first + 4 * k;
        if (at >= P.b0 && at + 4 <= P.b1) *(device uint2 *)(out + 2 * (at - P.b0)) = v[k];
    }
}

struct Wide32 {};

template <class E> struct Store {
    static void run(device uchar *out, thread const Params &P, ulong first, uint4 w) {
        store_block<E>(out, P, first, w);
    }
};
template <> struct Store<Wide32> {
    static void run(device uchar *out, thread const Params &P, ulong first, uint4 w) {
        store_block_wide(out, P, first, w);
    }
};

template <class E>
static void fill(thread const Params &P, device uchar *out, uint tid) {
    ulong c = 8ul * P.g0 + tid, g = c >> 3, lane = c & 7;
    ulong r0 = P.b0 >> 7, r1 = (P.b1 - 1) >> 7, row = g * P.K;
    if (g > r1 / P.K) return;
    State s = F_keyed(P.key, c, DOMAIN_STREAM, AUX_STREAM);
    uint j0 = row < r0 ? uint(r0 - row) : 0u;
    uint j1 = uint(min(r1 - row, ulong(P.K - 1)));
    for (uint j = 0; j <= j1; j++) {
        s = T(s);
        if (j >= j0) Store<E>::run(out, P, (row + j) * 128 + lane * 16, s.o);
    }
}


// ---- Normals --------------------------------------------------------------------------------
//
// Pair j is elements 2j and 2j + 1 from the f32 draws s0 + 2j and s0 + 2j + 1, where s0 is the
// index of the fill's first draw. A block holds draws 4b to 4b + 3. With s0 even a block holds
// two whole pairs. With s0 odd its second pair ends in the next block, which is the next lane at
// the same step, or for lane 7 lane 0 at the next step.

// Elements e and e + 1, or only e when e + 1 is past the fill.
static inline void store_pair(device float *out, thread const Params &P, ulong e, float2 z) {
    if (e + 1 >= P.n) {
        out[e] = z.x;
    } else if (((P.out_off + 4 * e) & 7) == 0) {
        *(device float2 *)(out + e) = z;
    } else {
        out[e] = z.x;
        out[e + 1] = z.y;
    }
}

// s0 even. The step bounds are computed once, as in `fill`.
static void fill_normal(thread const Params &P, device float *out, uint tid) {
    ulong c = 8ul * P.g0 + tid, g = c >> 3, lane = c & 7, row = g * P.K;
    ulong last = P.s0 + 2 * ((P.n + 1) / 2) - 1; // the last draw
    ulong r0 = P.s0 >> 5, r1 = last >> 5;
    if (row > r1) return;
    State s = F_keyed(P.key, c, DOMAIN_STREAM, AUX_STREAM);
    uint j0 = row < r0 ? uint(r0 - row) : 0u;
    uint j1 = uint(min(r1 - row, ulong(P.K - 1)));
    for (uint j = 0; j <= j1; j++) {
        s = T(s);
        if (j < j0) continue;
        // e + 2 for the element e of the block's first draw. It wraps for blocks before the
        // fill, so that every bound below fails for them.
        ulong e2 = 32 * (row + j) + 4 * lane + 2 - P.s0;
        float4 u = to_f32(s.o);
        if (e2 >= 2 && e2 + 2 <= P.n && ((P.out_off + 4 * e2 - 8) & 15) == 0) {
            *(device float4 *)(out + e2 - 2) = float4(normal_pair(u.x, u.y), normal_pair(u.z, u.w));
            continue;
        }
        if (e2 - 2 < P.n) store_pair(out, P, e2 - 2, normal_pair(u.x, u.y));
        if (e2 < P.n) store_pair(out, P, e2, normal_pair(u.z, u.w));
    }
}

// s0 odd. Neighbours swap word 0 through threadgroup memory, so every thread of a threadgroup
// runs the same steps and none returns early. A group's last pair needs the first block of the
// next group, which the next group shares, or for the last group of a threadgroup computes.
// The caller passes threadgroup arrays of THREADS and GROUPS words, since only a kernel can
// declare them.
static void fill_normal_odd(thread const Params &P, device float *out, uint tid, uint lt, uint tg,
                            threadgroup uint *word0, threadgroup uint *first0) {
    ulong c = 8ul * P.g0 + tid, g = c >> 3, lane = c & 7, row = g * P.K;
    ulong r1 = (P.s0 + 2 * ((P.n + 1) / 2) - 1) >> 5;
    ulong first_row = (P.g0 + ulong(tg) * GROUPS) * P.K;
    uint steps = uint(min(r1 - first_row, ulong(P.K - 1))) + 1;
    State s = F_keyed(P.key, c, DOMAIN_STREAM, AUX_STREAM);
    float held = 0.0f;
    ulong eheld = ~0ul; // lane 7: the pair that waits for the next step's word 0
    for (uint j = 0; j < steps; j++) {
        s = T(s);
        word0[lt] = s.o.x;
        if (j == 0 && lane == 0) first0[lt >> 3] = s.o.x;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (eheld < P.n) store_pair(out, P, eheld, normal_pair(held, to_f32(word0[lt - 7])));
        eheld = ~0ul;
        // The element of draw 4b + 1. It wraps for blocks before the fill.
        ulong e = 32 * (row + j) + 4 * lane + 1 - P.s0;
        float4 u = to_f32(s.o);
        if (e < P.n) store_pair(out, P, e, normal_pair(u.y, u.z));
        if (e + 2 < P.n) {
            if (lane < 7) store_pair(out, P, e + 2, normal_pair(u.w, to_f32(word0[lt + 1])));
            else { held = u.w; eheld = e + 2; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // A pair still held came from step K - 1: an earlier stop leaves its second draw past the fill.
    if (eheld < P.n) {
        uint gi = lt >> 3;
        uint x = gi + 1 < GROUPS ? first0[gi + 1] : block(P.key, 8 * (g + 1), 0).x;
        store_pair(out, P, eheld, normal_pair(held, to_f32(x)));
    }
}

} // namespace tandem

// ---- Kernels --------------------------------------------------------------------------------
//
// The Params in buffer 0, the output in buffer 1. Dispatch whole threadgroups of THREADS.

#ifndef TANDEM_NO_KERNELS

template <class E>
kernel void fill(constant tandem::Params &P [[buffer(0)]], device uchar *out [[buffer(1)]],
                 uint tid [[thread_position_in_grid]]) {
    tandem::Params p = P;
    tandem::fill<E>(p, out, tid);
}

template [[host_name("fill_u32")]] kernel void fill<tandem::U32>(constant tandem::Params &, device uchar *, uint);
template [[host_name("fill_f32")]] kernel void fill<tandem::F32>(constant tandem::Params &, device uchar *, uint);
template [[host_name("fill_exponential_f32")]] kernel void fill<tandem::Exponential32>(constant tandem::Params &, device uchar *, uint);
template [[host_name("fill_below32")]] kernel void fill<tandem::Below32>(constant tandem::Params &, device uchar *, uint);
template [[host_name("fill_below32_wide")]] kernel void fill<tandem::Wide32>(constant tandem::Params &, device uchar *, uint);
template [[host_name("fill_below64")]] kernel void fill<tandem::Below64>(constant tandem::Params &, device uchar *, uint);

kernel void fill_normal_f32(constant tandem::Params &P [[buffer(0)]], device float *out [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    tandem::Params p = P;
    tandem::fill_normal(p, out, tid);
}

kernel void fill_normal_f32_odd(constant tandem::Params &P [[buffer(0)]], device float *out [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lt [[thread_index_in_threadgroup]],
                                uint tg [[threadgroup_position_in_grid]]) {
    threadgroup uint word0[tandem::THREADS];
    threadgroup uint first0[tandem::GROUPS];
    tandem::Params p = P;
    tandem::fill_normal_odd(p, out, tid, lt, tg, word0, first0);
}

#endif
