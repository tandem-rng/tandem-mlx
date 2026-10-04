# API

## Use

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

## Reference

A key is a `(4,)` `uint32` array of key words. `split`, `purpose` and `fork` take any unsigned
64-bit index, scalar or array, and return keys of the index's shape plus `(4,)`. The chunk
length `K` changes only the stream layout, so every draw function takes `chunk_length`, 32 by
default, and keys do not carry it.

Every draw function reads the stream from the bit `position`, 0 by default, after aligning it
to the draw width as the specification says. The `stream`, `stream_normal`, `stream_randint`
and `stream_exponential` forms return the draws and the position after them.

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

## Parallel use

Element `i` of a fill is draw `i`, so fills at offsets reproduce one whole fill, for any split
of the work, as
[Appendix B](https://github.com/tandem-rng/spec/blob/main/SPEC.md#appendix-b-parallel-decomposition-non-normative)
of the specification shows.
