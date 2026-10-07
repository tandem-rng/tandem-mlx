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
- normal fills from even and odd draws, over many groups and every end lane, against Box-Muller
  applied to the uniform fill;
- that a bounded fill cut at any element equals the whole fill, rejections included;
- raw moments 1 to 4 and the Kolmogorov-Smirnov distance of 10^7 normals and 10^7
  exponentials;
- that `tandem.metal` is tandem-metal's file, by its SHA-256;
- with tandem-numpy installed, words, uniforms, bounded integers, normals and exponentials
  against its C fills, bit for bit.

`tests/test_conformance.py` reads the spec's conformance files and checks the items of its
`conformance/CHECKLIST.md` at 2a4bd08:

- every bounded, Float32 normal, Float32 exponential and weighted choice case, values and end
  positions, whole, cut at elements 1, 7, 20, 21 and `n - 1` (2, 8, 20 and the last even element
  for normals), and one element at a time;
- the scalar bounded cases, by Lemire's loop over the port's plain draws;
- the fallback index of rejected draws, the width that follows the range, the empty fills, odd
  `n` and the normal pairs;
- the choice tables and rejected weights;
- the SHA-256 of the stream dumps and of the port's fills of them;
- reads across blocks, rows and chunks, a draw at 2^63 - 1, and fills that would reach 2^64.

Metal has no double type, so the Float64 cases and the long outputs of tandem-c's dump tools are
not run, and MLX has no complex draws.

## Fixtures

- `tests/vectors.json` is a copy of the spec repository's file.
- `tests/conformance` holds the spec's `conformance/*.json` at commit `2a4bd08`.
- `tests/data` holds the stream dumps of tandem-c.
- `tests/cross_port.json` comes from tandem-jax.

## CI

CI runs `pytest` on macos-15 with Python 3.13 and on macos-latest with Python 3.14, with
tandem-numpy built from source for the cross-check. A drift check fails when
`tests/vectors.json` differs from the spec repository's file, or `tests/conformance` from the
spec's `conformance` directory at `2a4bd08`.
