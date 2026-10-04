<p align="center"><img src="assets/lockup.png" width="560" alt="tandem rng .mlx"></p>

# tandem-mlx

[![CI](https://github.com/tandem-rng/tandem-mlx/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/tandem-rng/tandem-mlx/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](LICENSE)

[MLX](https://github.com/ml-explore/mlx) binding of [Tandem8x32](https://github.com/tandem-rng/spec),
a noncryptographic pseudorandom number generator. Its draws run as Metal kernels on Apple silicon
GPUs, bit for bit with the specification and tandem-c, near the GPU's write bandwidth.

Install with pip. It needs macOS 14 or later on Apple silicon, Python 3.13 or later and MLX 0.32
or later. The shader is tandem-metal's `tandem.metal` at `6bd3824`, vendored unchanged.

```sh
pip install .
pixi run test
```

```python
import mlx.core as mx
import tandem_mlx as tm

key = tm.key(42)                                   # the spec's generator for seed 42, a (4,) uint32 array
x, pos = tm.stream(key, 0, 2**20, mx.float32)      # draws and the position after them
kid = tm.split(key, 7)                             # the spec's split child 7
z = tm.normal(kid, (1000,))                        # Box-Muller normals, Appendix A
```

See [docs/notes.md](docs/notes.md) for the API, the float64 rules, tests and speed.

Portions of the code were generated with the assistance of LLMs.

[Documentation](docs/notes.md) · [Apache 2.0 license](LICENSE)
