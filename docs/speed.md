# Speed

`pixi run bench` produces the figures.

## GPU

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
