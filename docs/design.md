# Design

## Metal source

`src/tandem_mlx/tandem.metal` is the shader source of
[tandem-metal](https://github.com/tandem-rng/tandem-metal), commit `3c0b1be`, unchanged. A test
checks its hash. It holds the building blocks, the derived draws and the fills as functions in
namespace `tandem`, and kernels for a Metal host, which tandem-mlx leaves out by defining
`TANDEM_NO_KERNELS`. `tm.metal_source()` returns that header, so other `mx.fast.metal_kernel`
code can use it.

## Fills

One thread runs one chunk for its `K` steps, and the eight threads of a group store one
contiguous 128-byte row per step. Normals from an odd first draw pair the last word of a block
with the first word of the next block through threadgroup memory.

## Normals and exponentials

The normals use the polynomial logarithm, sine and cosine of tandem-c, and the exponentials
tandem-c's two-part `-log`, within 0.58 ulp, all with explicit `fma` and
`metal::precise::divide` and `metal::precise::sqrt`. They are bit for bit
equal to tandem-c under every Metal math mode. Plain `sqrt` is not correctly rounded even in
Metal's safe mode, and plain `/` is not in fast mode. MLX compiles custom kernels in safe mode
by default, and tandem-mlx passes `compile_options={"math_mode": "safe"}`, which needs MLX 0.32.
