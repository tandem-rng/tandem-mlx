// Tandem8x32 in the Metal Shading Language: the building blocks of
// https://github.com/tandem-rng/spec, the derived draws of its Appendix A, and the fill loop.
// Copyright 2026 Jessica Cox. Apache License 2.0, see LICENSE.
//
// Plain inline functions with no kernels and no buffers, so that MLX can take the file as the
// header of `mx.fast.metal_kernel` and a Metal host can wrap the same functions in kernels.
//
// The normals and exponentials equal tandem-c bit for bit under every math mode, fast included:
// their division and square root are the precise:: functions, since plain `sqrt` is not
// correctly rounded even in MTLMathModeSafe and plain `/` is not under fast math, and every
// multiply-add that matters is an explicit fma.

#include <metal_stdlib>

namespace tandem {

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
inline State T(State s) {
    uint m0 = s.h.x | 1u, m1 = s.h.y | 1u;
    uint lo0 = s.o.x * m0, hi0 = metal::mulhi(s.o.x, m0);
    uint lo1 = s.o.z * m1, hi1 = metal::mulhi(s.o.z, m1);
    uint4 n = uint4(s.o.y ^ hi1 ^ lo1, metal::rotate(lo1, 16u) ^ s.h.z, s.o.w ^ hi0 ^ lo0,
                    metal::rotate(lo0, 16u) ^ s.h.w);
    uint4 h = s.h;
    h.x ^= metal::rotate(h.y, 7u);
    h.y ^= metal::rotate(h.z, 13u);
    h.z ^= metal::rotate(h.w, 22u);
    h.w ^= metal::rotate(h.x, 3u);
    h.x = (h.x + CLOCK_WEYL) ^ n.x;
    return {n, h};
}

inline State round(State s, uint rc) {
    State r = T(s);
    return {r.h, uint4(r.o.x ^ rc, r.o.yzw)};
}

// The seeding function F, written out: a constant array indexed in a loop can land in
// per-thread memory.
inline State F(State s) {
    s = round(s, 0xd17cc1b7u);
    s = round(s, 0xa7220a94u);
    s = round(s, 0xfe13abe8u);
    s = round(s, 0xfa9a6ee0u);
    s = round(s, 0xedb14accu);
    s = round(s, 0x9e21c820u);
    s = round(s, 0xff28b1d5u);
    return round(s, 0xef5de2b0u);
}

inline State F_keyed(uint4 key, ulong counter, uint domain, uint aux) {
    return F({uint4(uint(counter), uint(counter >> 32), domain, aux), key});
}

// Block B(c, j): the exposed half of chunk c after j + 1 steps.
inline uint4 block(uint4 key, ulong c, uint j) {
    State s = F_keyed(key, c, DOMAIN_STREAM, AUX_STREAM);
    for (uint i = 0; i <= j; i++) s = T(s);
    return s.o;
}

inline uint4 split_key(uint4 key, ulong index) {
    State s = F_keyed(key, index >> 1, DOMAIN_SPLIT, 0u);
    return (index & 1ul) ? s.h : s.o;
}

inline uint4 sub_key(uint4 key, ulong purpose) {
    return F_keyed(key, purpose, DOMAIN_FOLD, 0u).o;
}

// Word j of the stream of `key` from position 0, for chunk length K.
template <uint K>
inline uint stream_word(uint4 key, ulong j) {
    ulong b = j >> 2, row = b >> 3;
    return block(key, (row / K) * 8ul + (b & 7ul), uint(row % K))[uint(j & 3ul)];
}

// ---- Appendix A: bounded integers ------------------------------------------------------------
//
// Lemire's method. A rejected draw of a fill retries on the stream of split(g) of
// purpose(P_w) of the fill's key, g the draw's index in the key's stream.

template <uint K>
inline uint below32(uint x, uint range, uint4 key, ulong g) {
    if (range == 0u) return 0u;
    if (x * range >= range) return metal::mulhi(x, range);
    uint t = (0u - range) % range;
    if (x * range >= t) return metal::mulhi(x, range);
    uint4 k2 = split_key(sub_key(key, PURPOSE_BELOW32), g);
    for (ulong j = 0;; j++) {
        uint y = stream_word<K>(k2, j);
        if (y * range >= t) return metal::mulhi(y, range);
    }
}

template <uint K>
inline ulong below64(ulong x, ulong range, uint4 key, ulong g) {
    if (range == 0ul) return 0ul;
    if (x * range >= range) return metal::mulhi(x, range);
    ulong t = (0ul - range) % range;
    if (x * range >= t) return metal::mulhi(x, range);
    uint4 k2 = split_key(sub_key(key, PURPOSE_BELOW64), g);
    for (ulong j = 0;; j++) {
        ulong y = ulong(stream_word<K>(k2, 2ul * j)) | (ulong(stream_word<K>(k2, 2ul * j + 1ul)) << 32);
        if (y * range >= t) return metal::mulhi(y, range);
    }
}

// ---- Appendix A: normals and exponentials, the arithmetic of tandem-c --------------------------

// The spec's Float32 mapping. The shift leaves 24 bits, so both steps are exact.
inline float to_f32(uint x) { return float(x >> 8) * 0x1p-24f; }

// -2 ln x for x in (0, 1]: x = mant 2^k with mant in [sqrt(1/2), sqrt(2)) from the bits, then
// -2 ln x = 2 nk ln 2 - 4 s p with s = (mant - 1) / (mant + 1) and p a series in s^2.
inline float neg2_log_f32(float x) {
    uint ix = as_type<uint>(x) + 0x004afb0du;
    float nk = float(127 - int(ix >> 23));
    float mant = as_type<float>((ix & 0x007fffffu) + 0x3f3504f3u);
    float s = metal::precise::divide(mant - 1.0f, mant + 1.0f), zz = s * s;
    float p = metal::fma(zz, metal::fma(zz, metal::fma(zz, 0.14275366f, 0.20000061f), 0.33333334f), 1.0f);
    return metal::fma(nk, 2.857213530660374e-06f, metal::fma(nk, 1.38629150390625f, (s * -4.0f) * p));
}

inline float exponential_f32(float u) { return 0.5f * neg2_log_f32(1.0f - u); }

// Box-Muller pair (r cos 2 pi b, r sin 2 pi b), r = sqrt(-2 ln(1 - a)). b - q/4 for the nearest
// quarter turn q is exact, so short series on [-pi/4, pi/4] give cos and sin, and the quarter
// turn is a swap and sign change.
inline float2 normal_pair_f32(float a, float b) {
    float r = metal::precise::sqrt(neg2_log_f32(1.0f - a));
    int q = int(b * 4.0f + 0.5f);
    float f = metal::fma(-float(q), 0.25f, b);
    // 2 pi as a float pair, so the angle is good to the last bit of the float.
    float th = metal::fma(f, -1.7484555e-7f, f * 6.2831855f), w = th * th;
    float hs = metal::fma(w, metal::fma(w, metal::fma(w, 2.72499e-06f, -0.00019840087f), 0.008333332f), -0.16666667f);
    float hc = metal::fma(w, metal::fma(w, metal::fma(w, 2.4463761e-05f, -0.0013887589f), 0.04166665f), -0.5f);
    float sn = th * metal::fma(w, hs, 1.0f), cs = metal::fma(w, hc, 1.0f);
    uint qu = uint(q), sm = 0u - (qu & 1u);
    uint sb = as_type<uint>(sn), cb = as_type<uint>(cs);
    uint xb = ((sb & sm) | (cb & ~sm)) ^ (((qu + 1u) << 30) & 0x80000000u);
    uint yb = ((cb & sm) | (sb & ~sm)) ^ ((qu << 30) & 0x80000000u);
    return r * float2(as_type<float>(xb), as_type<float>(yb));
}

// ---- Fill loop ---------------------------------------------------------------------------------
//
// One thread per chunk. Thread t runs chunk 8 g + lane of group g = g_first + t / 8 for its K
// steps, and step j yields stream block (g K + j) 8 + lane. The eight lanes of a group write one
// contiguous 128-byte row per step, so the stores coalesce without staging.

enum Kind : int {
    BITS32 = 0,     // the 32-bit draws
    BITS64 = 1,     // the 64-bit draws
    UNIFORM32 = 2,  // Float32 uniforms
    BELOW32 = 3,    // lo + Lemire on 32-bit draws, 32-bit output
    BELOW32_64 = 4, // lo + Lemire on 32-bit draws, 64-bit output
    BELOW64 = 5,    // lo + Lemire on 64-bit draws
    NORMAL32 = 6,   // Float32 normals, pairs inside a block: an even first draw
    EXPONENTIAL32 = 7,
    NORMAL32_ODD = 8, // Float32 normals from an odd first draw
};

struct FillParams {
    uint4 key;
    ulong g_first; // first group of chunks
    uint base;     // the fill's first block, counted from the first block of group g_first
    uint skip;     // draws of that block before element 0
    uint n;        // elements
    ulong d0;      // index of the draw of element 0 in the key's stream
    ulong range;   // bounded draws: values on [0, range) ...
    ulong lo;      // ... plus lo, modulo 2^64
};

// The key's four words, and 11 words g_first, base, skip, n, d0, range, lo, each 64-bit value
// low word first. Templates, since a host may bind either in the constant or the device space.
template <typename KeyPtr, typename ParamPtr>
inline FillParams unpack(KeyPtr k, ParamPtr p) {
    return {uint4(k[0], k[1], k[2], k[3]), ulong(p[0]) | (ulong(p[1]) << 32), p[2], p[3], p[4],
            ulong(p[5]) | (ulong(p[6]) << 32), ulong(p[7]) | (ulong(p[8]) << 32),
            ulong(p[9]) | (ulong(p[10]) << 32)};
}

// A range of 2^32 on 32-bit draws takes every draw as it is.
template <uint K>
inline ulong bounded32(uint x, ulong g, thread const FillParams &p) {
    return ulong(p.range >> 32 ? x : below32<K>(x, uint(p.range), p.key, g)) + p.lo;
}

template <int KIND, uint K>
inline uint map32(uint x, ulong g, thread const FillParams &p) {
    if (KIND == UNIFORM32) return as_type<uint>(to_f32(x));
    if (KIND == EXPONENTIAL32) return as_type<uint>(exponential_f32(to_f32(x)));
    if (KIND == BELOW32) return uint(bounded32<K>(x, g, p));
    return x;
}

template <int KIND, uint K>
inline ulong map64(ulong x, ulong g, thread const FillParams &p) {
    if (KIND == BELOW64) return below64<K>(x, p.range, p.key, g) + p.lo;
    return x;
}

// Elements e .. e + 3 from the four 32-bit draws x, 32-bit outputs.
template <int KIND, uint K>
inline void store32(device uint *out, uint4 x, int e, thread const FillParams &p) {
    int n = int(p.n);
    if (e >= 0 && e + 4 <= n) {
        ulong g = p.d0 + ulong(e);
        uint4 v;
        if (KIND == NORMAL32) {
            v = as_type<uint4>(float4(normal_pair_f32(to_f32(x.x), to_f32(x.y)),
                                             normal_pair_f32(to_f32(x.z), to_f32(x.w))));
        } else {
            v = uint4(map32<KIND, K>(x.x, g, p), map32<KIND, K>(x.y, g + 1, p),
                      map32<KIND, K>(x.z, g + 2, p), map32<KIND, K>(x.w, g + 3, p));
        }
        if (p.skip == 0u) {
            *(device uint4 *)(out + e) = v;
        } else {
            out[e] = v.x, out[e + 1] = v.y, out[e + 2] = v.z, out[e + 3] = v.w;
        }
        return;
    }
    if (KIND == NORMAL32) {
        // An even first draw keeps both pairs inside the block. An odd n drops the last sin half.
        float2 z0 = normal_pair_f32(to_f32(x.x), to_f32(x.y));
        float2 z1 = normal_pair_f32(to_f32(x.z), to_f32(x.w));
        float z[4] = {z0.x, z0.y, z1.x, z1.y};
        for (int k = 0; k < 4; k++)
            if (e + k >= 0 && e + k < n) out[e + k] = as_type<uint>(z[k]);
        return;
    }
    for (int k = 0; k < 4; k++)
        if (e + k >= 0 && e + k < n) out[e + k] = map32<KIND, K>(x[k], p.d0 + ulong(e + k), p);
}

// Elements e .. e + 3 from the four 32-bit draws x, 64-bit outputs as word pairs.
template <uint K>
inline void store32_wide(device uint *out, uint4 x, int e, thread const FillParams &p) {
    int n = int(p.n);
    for (int k = 0; k < 4; k++)
        if (e + k >= 0 && e + k < n) {
            ulong v = bounded32<K>(x[k], p.d0 + ulong(e + k), p);
            *(device uint2 *)(out + 2 * (e + k)) = uint2(uint(v), uint(v >> 32));
        }
}

// Elements e and e + 1 from the two 64-bit draws in x, as word pairs.
template <int KIND, uint K>
inline void store64(device uint *out, uint4 x, int e, thread const FillParams &p) {
    int n = int(p.n);
    ulong a = ulong(x.x) | (ulong(x.y) << 32), b = ulong(x.z) | (ulong(x.w) << 32);
    if (e >= 0 && e + 2 <= n) {
        ulong g = p.d0 + ulong(e);
        a = map64<KIND, K>(a, g, p), b = map64<KIND, K>(b, g + 1, p);
        uint4 v = uint4(uint(a), uint(a >> 32), uint(b), uint(b >> 32));
        if (p.skip == 0u) {
            *(device uint4 *)(out + 2 * e) = v;
        } else {
            *(device uint2 *)(out + 2 * e) = v.xy, *(device uint2 *)(out + 2 * e + 2) = v.zw;
        }
        return;
    }
    if (e >= 0 && e < n) {
        a = map64<KIND, K>(a, p.d0 + ulong(e), p);
        *(device uint2 *)(out + 2 * e) = uint2(uint(a), uint(a >> 32));
    }
    if (e + 1 >= 0 && e + 1 < n) {
        b = map64<KIND, K>(b, p.d0 + ulong(e + 1), p);
        *(device uint2 *)(out + 2 * e + 2) = uint2(uint(b), uint(b >> 32));
    }
}

// Elements i and i + 1 from the Box-Muller pair of the uniform words a and b.
inline void store_pair(device uint *out, uint a, uint b, int i, int n) {
    if (i + 1 < 0 || i >= n) return;
    float2 z = normal_pair_f32(to_f32(a), to_f32(b));
    if (i >= 0) out[i] = as_type<uint>(z.x);
    if (i + 1 < n) out[i + 1] = as_type<uint>(z.y);
}

// Normals from an odd first draw. Words 1 and 2 of a block form a pair, and word 3 pairs with
// word 0 of the next block. Lane l < 7 reads that word of lane l + 1 by a SIMD shuffle in the
// same step. Lane 7 reads it from lane 0 one step later and computes it after the last step.
// The eight lanes of a group are adjacent SIMD lanes and run the same steps.
template <uint K>
inline void fill_normal_odd(device uint *out, thread const FillParams &p, uint t) {
    uint gi = t >> 3, lane = t & 7u;
    ulong g = p.g_first + gi;
    State s = F_keyed(p.key, g * 8ul + lane, DOMAIN_STREAM, AUX_STREAM);
    int n = int(p.n);
    int e = (int((gi * K) * 8u + lane) - int(p.base)) * 4 - int(p.skip);
    // The group's first element of the step. Word 3 of lane 7 one step back is element row - 1.
    int row = e - 4 * int(lane);
    uint w3 = 0u, j = 0u;
    for (; j < K && row <= n; j++, e += 32, row += 32) {
        s = T(s);
        uint next = metal::simd_shuffle_down(s.o.x, 1u);
        uint first = metal::simd_shuffle_up(s.o.x, 7u);
        if (lane == 7u && j > 0u) store_pair(out, w3, first, row - 1, n);
        store_pair(out, s.o.y, s.o.z, e + 1, n);
        if (lane < 7u) store_pair(out, s.o.w, next, e + 3, n);
        w3 = s.o.w;
    }
    if (lane == 7u && j == K && row - 1 < n) store_pair(out, w3, block(p.key, (g + 1ul) * 8ul, 0u).x, row - 1, n);
}

template <int KIND, uint K>
inline void fill(device uint *out, thread const FillParams &p, uint t) {
    if (KIND == NORMAL32_ODD) {
        fill_normal_odd<K>(out, p, t);
        return;
    }
    constexpr int PER = (KIND == BITS64 || KIND == BELOW64) ? 2 : 4;
    uint gi = t >> 3, lane = t & 7u;
    State s = F_keyed(p.key, (p.g_first + gi) * 8ul + lane, DOMAIN_STREAM, AUX_STREAM);
    // First element of this thread's block at step j, relative to the fill.
    int e = (int((gi * K) * 8u + lane) - int(p.base)) * PER - int(p.skip);
    for (uint j = 0; j < K && e < int(p.n); j++, e += 8 * PER) {
        s = T(s);
        if (e + PER <= 0) continue;
        if (PER == 2) {
            store64<KIND, K>(out, s.o, e, p);
        } else if (KIND == BELOW32_64) {
            store32_wide<K>(out, s.o, e, p);
        } else {
            store32<KIND, K>(out, s.o, e, p);
        }
    }
}

} // namespace tandem
