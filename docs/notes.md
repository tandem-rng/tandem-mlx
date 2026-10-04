# Notes

## API

```python
import mlx.core as mx
import tandem_mlx as tm

key = tm.key(42)                                   # the spec's generator for seed 42, a (4,) uint32 array
u = tm.uniform(key, (1000,))                       # float32 uniforms, the spec's mapping
z = tm.normal(key, (1000,))                        # Box-Muller normals, Appendix A
e = tm.exponential(key, (1000,))                   # -log(1 - u), Appendix A
r = tm.randint(key, (1000,), 0, 6)                 # Lemire bounded integers, Appendix A
w = tm.bits(key, (16,), mx.uint32)                 # the stream words from position 0

x, pos = tm.stream(key, 0, 2**20, mx.float32)      # draws and the position after them
y, pos = tm.stream(key, pos, 100, mx.uint8)        # continue there, aligned per the spec
z, pos = tm.stream_normal(key, pos, 1000)          # also stream_randint, stream_exponential
u = tm.uniform(key, (1000,), position=2**40)       # any draw function reads from any bit position

kid = tm.split(key, 7)                             # the spec's split child 7
kids = tm.split(key, mx.arange(4))                 # children 0 to 3, shape (4, 4)
sub = tm.purpose(key, 3)                           # the spec's purpose child
kids, pos = tm.fork(key, pos, 4)                   # the spec's fork at the current block
u8 = tm.uniform(key, (1000,), chunk_length=8)      # the Tandem8x32-K8 variant
```

A key is a `(4,)` `uint32` array of key words. `split`, `purpose` and `fork` take any unsigned
64-bit index, scalar or array, and return keys of the index's shape plus `(4,)`. The chunk
length `K` changes only the stream layout, so every draw function takes `chunk_length`, 32 by
default, and keys do not carry it.

Every draw function reads the stream from the bit `position`, 0 by default, after aligning it to
the draw width as the specification says. The `stream`, `stream_normal`, `stream_randint` and
`stream_exponential` forms return the draws and the position after them. Element `i` of a fill
is draw `i`, so fills at offsets reproduce one whole fill, for any split of the work, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.

- `stream(key, position, n, dtype)` and `bits(key, shape, dtype, position)`: unsigned integers of
  8 to 64 bits are the raw draws, signed ones reinterpret them.
- `uniform(key, shape, dtype, position, low=0, high=1)`: `float32` as `(raw >> 8) * 2**-24`,
  `float16` as `(raw >> 5) * 2**-11` and `float64` as `(raw >> 11) * 2**-53`, the spec's exact
  mappings.
- `randint(key, shape, low, high, dtype, position, width=None)`: Lemire's method on draw `i` for
  element `i`. The draw width follows the range, 32 bits up to `2**32` and 64 bits above, so the
  dtype does not change the values, and `width` names it as the `u32` and `u64` fills of the C
  and CUDA ports do. A rejected draw retries on the fallback stream `split(g)` of
  `purpose(0x424c573332)` (`0x424c573634` for 64 bits), with `g` the index of the draw in the
  key's stream, so a fill cut at any element boundary equals the whole fill. `high <= low`
  gives `low`.
- `normal(key, shape, dtype, position, loc=0, scale=1)`: pair `j` is elements `2j` and `2j + 1`,
  `r cos 2 pi b` and `r sin 2 pi b` with `r = sqrt(-2 log(1 - a))`, from uniform draws `2j` and
  `2j + 1`. A fill consumes `2 ceil(n / 2)` draws.
- `exponential(key, shape, dtype, position)`: `-log(1 - u)` from one uniform draw each.

Derived fills of zero elements leave the position as it is. `mx.random.uniform`, `normal` and
`randint` use MLX's own threefry key and give different values.

**float64.** Metal has no double type. `uniform` and `stream` take `float64`: the kernel writes
the 64-bit draws and the CPU stream applies the exact mapping, so the result is an array for the
CPU stream, `mx.cpu`. Normals and exponentials need double-precision arithmetic, so `float64`
raises `TypeError`. Use [tandem-numpy](https://github.com/tandem-rng/tandem-numpy) for them,
which gives the same values as every other port.

## The Metal source

`src/tandem_mlx/tandem.metal` is the shader source of
[tandem-metal](https://github.com/tandem-rng/tandem-metal), commit `6bd3824`, unchanged. A test
checks its hash. It holds the building blocks, the derived draws and the fills as functions in
namespace `tandem`, and kernels for a Metal host, which tandem-mlx leaves out by defining
`TANDEM_NO_KERNELS`. `tm.metal_source()` returns that header, so other `mx.fast.metal_kernel`
code can use it. One thread runs one chunk for its `K` steps, and the eight threads of a group
store one contiguous 128-byte row per step. Normals from an odd first draw pair the last word of
a block with the first word of the next block through threadgroup memory.

The normals and exponentials use the polynomial logarithm, sine and cosine of tandem-c with
explicit `fma` and `metal::precise::divide` and `metal::precise::sqrt`. They are bit for bit
equal to tandem-c under every Metal math mode. Plain `sqrt` is not correctly rounded even in
Metal's safe mode, and plain `/` is not in fast mode. MLX compiles custom kernels in safe mode
by default, and tandem-mlx passes `compile_options={"math_mode": "safe"}`, which needs MLX 0.32.

## Install

```sh
pip install .
```

Needs macOS 14 or later on Apple silicon, Python 3.13 or later, and MLX 0.32 or later. For
development, `pixi run test` runs the tests. `pixi run -e cross test` also builds tandem-numpy
from its repository and checks that both give the same values.

## Tests

`tests/test_tandem.py` checks:

- every vector of the specification (`tests/vectors.json`, a copy of the spec repository's file,
  with a drift check in CI), including T and F of the Metal source on the vector states;
- the stream dumps of tandem-c in `tests/data` for `uint8`, `uint32`, `uint64`, `float16`,
  `float32`, `float64` and K = 8, from position 0 and from unaligned starts;
- blocks at rows past 2**50, where the chunk counter needs 64 bits, against a host evaluation of
  F and T for K = 1, 8, 32 and 65536;
- split, fork and purpose children for indices up to 2**64 - 1 against the C reference
  (`tests/cross_port.json`, from tandem-jax);
- bounded fills, f32 normals and f32 exponentials against the fixtures of tandem-c `b049384`
  and tandem-cuda `c5c5725`, bit for bit, in `tests/cross_derived.json`, which
  `tools/convert_c_fixtures.py` writes from their headers. These include the 28 bounded fills
  of tandem-cuda at starts 1 and 12345 with up to 41 rejected elements, the scalar bounded
  draws against Lemire's loop, and normal and exponential fills from even and odd draws;
- normal fills from even and odd draws, over many groups and every end lane, against Box-Muller
  applied to the uniform fill;
- that a bounded fill cut at any element equals the whole fill, rejections included, and that
  empty fills keep the position;
- raw moments 1 to 4 and the Kolmogorov-Smirnov distance of 10^7 normals and 10^7
  exponentials;
- that `tandem.metal` is tandem-metal's file, by its SHA-256;
- with tandem-numpy installed, words, uniforms, bounded integers, normals and exponentials
  against its C fills, bit for bit.

## Speed

Apple M4 Pro, MLX 0.32.3, `pixi run bench`. A run issues ten calls back to back and evaluates
them together. The table gives the best of five runs after a warm-up, in GiB/s of output.
`mx.random` uses its default threefry key.

| draw | log2 n | tandem_mlx | mx.random |
|---|---|---|---|
| uniform float32 | 24 | 160 | 20 |
| bits uint32 | 24 | 181 | - |
| randint int32 in [0, 1000) | 24 | 184 | 12 |
| normal float32 | 24 | 157 | 15 |
| normal float32, odd start | 24 | 149 | - |
| exponential float32 | 24 | 177 | - |
| uniform float32 | 26 | 181 | 20 |
| bits uint32 | 26 | 183 | - |
| randint int32 in [0, 1000) | 26 | 180 | 12 |
| normal float32 | 26 | 179 | 15 |
| normal float32, odd start | 26 | 159 | - |
| exponential float32 | 26 | 179 | - |

`mx.full` writes 187 GiB/s at 2^24 and 196 GiB/s at 2^26 on the same GPU, so the fills run near
the write bandwidth.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
