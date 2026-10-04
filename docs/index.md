# tandem-mlx

[MLX](https://github.com/ml-explore/mlx) binding of Tandem8x32. Its draws run as Metal kernels
on Apple silicon GPUs, bit for bit with the
[specification](https://github.com/tandem-rng/spec/blob/main/SPEC.md) and tandem-c, near the
GPU's write bandwidth.

- [API](api.md): the draw functions, keys, positions and the float64 rules.
- [Design](design.md): the vendored Metal source, its fills and its arithmetic.
- [Tests](tests.md): what the suite checks, the fixtures, and what CI runs.
- [Speed](speed.md): Apple M4 Pro figures against `mx.random`.

## Install

```sh
pip install .
```

Needs macOS 14 or later on Apple silicon, Python 3.13 or later, and MLX 0.32 or later. For
development, `pixi run test` runs the tests. `pixi run -e cross test` also builds tandem-numpy
from its repository and checks that both give the same values.

## AI assistance

This port was written with the help of large language models under human
direction. The design and the specification are human work, as is much of the
Julia implementation. The code is tested bit for bit against every vector of
the specification and against long stream dumps from the Julia implementation,
and every value must match. The output does not depend on who or what wrote the
code.
