# Tests

```sh
pixi run test
pixi run -e cross test    # also builds tandem-numpy from its repository and compares
```

## Suite

`tests/test_tandem.py` checks:

- every vector of the specification, including T and F of the Metal source on the vector states;
- the stream dumps of tandem-c in `tests/data` for `uint8`, `uint32`, `uint64`, `float16`,
  `float32`, `float64` and K = 8, from position 0 and from unaligned starts;
- blocks at rows past 2**50, where the chunk counter needs 64 bits, against a host evaluation of
  F and T for K = 1, 8, 32 and 65536;
- split, fork and purpose children for indices up to 2**64 - 1 against the C reference
  (`tests/cross_port.json`);
- bounded fills, f32 normals and f32 exponentials against the fixtures of tandem-c and
  tandem-cuda, bit for bit, in `tests/cross_derived.json`. These include the 28 bounded fills
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

## Fixtures

- `tests/vectors.json` is a copy of the spec repository's file.
- `tests/data` holds the stream dumps of tandem-c.
- `tests/cross_port.json` comes from tandem-jax.
- `tests/cross_derived.json` holds the fixtures of tandem-c `b049384` and tandem-cuda
  `c5c5725`, which `tools/convert_c_fixtures.py` writes from their headers.

## CI

CI runs `pytest` on macos-15 with Python 3.13 and on macos-latest with Python 3.14, with
tandem-numpy built from source for the cross-check. A drift check fails when
`tests/vectors.json` differs from the spec repository's file.
