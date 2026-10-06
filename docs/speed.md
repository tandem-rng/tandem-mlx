# Speed

`pixi run bench` produces the figures, in its own environment with PyObjC for the MPS baseline.

## GPU

Apple M4 Pro, MLX 0.32.3, one session. A run issues ten calls back to back and evaluates them
together. The table gives the best of five runs after a warm-up, in GiB/s of output.
`mx.random` uses its default threefry key. The baseline is MPSMatrixRandomPhilox, Philox4x32-10
from Metal Performance Shaders, timed the same way with ten fills in one command buffer. MPS has
no bounded integers and no exponentials, so those rows compare with its uint32 words or float32
uniforms, marked *.

| draw | log2 n | tandem_mlx | MPS Philox | mx.random |
|---|---|---|---|---|
| uniform float32 | 24 | 152 | 162 | 20 |
| bits uint32 | 24 | 165 | 165 | - |
| randint int32 in [0, 1000) | 24 | 173 | 166* | 12 |
| normal float32 | 24 | 170 | 105 | 14 |
| normal float32, odd start | 24 | 152 | 105 | - |
| exponential float32 | 24 | 174 | 163* | - |
| uniform float32 | 26 | 173 | 145 | 20 |
| bits uint32 | 26 | 164 | 146 | - |
| randint int32 in [0, 1000) | 26 | 145 | 145* | 11 |
| normal float32 | 26 | 159 | 108 | 15 |
| normal float32, odd start | 26 | 159 | 109 | - |
| exponential float32 | 26 | 178 | 164* | - |

Tandem and MPS Philox both run near the GPU's integer throughput on the plain draws. The MPS
normals run at two thirds of its uniform rate, while the Tandem Box-Muller pass adds little to
the fill.
