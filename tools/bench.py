"""GiB/s of output on the default GPU at 2^24 and 2^26 elements, against mx.random and
MPSMatrixRandomPhilox, as a markdown table. A run issues ten calls back to back and evaluates
them together. The table gives the best of five runs after a warm-up. Run it on a quiet machine.
"""

import time

import Metal
import MetalPerformanceShaders as MPS
import mlx.core as mx

import tandem_mlx as tm

KEY = tm.key(42)
MKEY = mx.random.key(42)

# MPS has Philox for uint32 words, float32 uniforms and float32 normals. A draw with no MPS
# counterpart gets the nearest of these, marked in the table.
DEVICE = Metal.MTLCreateSystemDefaultDevice()
QUEUE = DEVICE.newCommandQueue()
DISTRIBUTIONS = {
    "words": (MPS.MPSDataTypeUInt32, MPS.MPSMatrixRandomDistributionDescriptor.defaultDistributionDescriptor()),
    "uniform": (
        MPS.MPSDataTypeFloat32,
        MPS.MPSMatrixRandomDistributionDescriptor.uniformDistributionDescriptorWithMinimum_maximum_(0.0, 1.0),
    ),
    "normal": (
        MPS.MPSDataTypeFloat32,
        MPS.MPSMatrixRandomDistributionDescriptor.normalDistributionDescriptorWithMean_standardDeviation_(0.0, 1.0),
    ),
}


def philox(kind):
    dtype, desc = DISTRIBUTIONS[kind]
    kernel = MPS.MPSMatrixRandomPhilox.alloc().initWithDevice_destinationDataType_seed_distributionDescriptor_(
        DEVICE, dtype, 42, desc
    )
    vectors = {}

    def run(n, calls):
        if n not in vectors:
            buf = DEVICE.newBufferWithLength_options_(4 * n, Metal.MTLResourceStorageModePrivate)
            desc = MPS.MPSVectorDescriptor.vectorDescriptorWithLength_dataType_(n, dtype)
            vectors[n] = MPS.MPSVector.alloc().initWithBuffer_descriptor_(buf, desc)
        cb = QUEUE.commandBuffer()
        for _ in range(calls):
            kernel.encodeToCommandBuffer_destinationVector_(cb, vectors[n])
        cb.commit()
        cb.waitUntilCompleted()

    return run


def mlx_calls(f):
    return lambda n, calls: mx.eval(*[f(n) for _ in range(calls)])


# (label, bytes per element, tandem_mlx, mx.random or None, MPS kind, whether MPS draws differ)
ROWS = [
    ("uniform float32", 4, lambda n: tm.uniform(KEY, n), lambda n: mx.random.uniform(shape=(n,), key=MKEY), "uniform", False),
    ("bits uint32", 4, lambda n: tm.bits(KEY, n), None, "words", False),
    (
        "randint int32 in [0, 1000)",
        4,
        lambda n: tm.randint(KEY, n, 0, 1000),
        lambda n: mx.random.randint(0, 1000, (n,), key=MKEY),
        "words",
        True,
    ),
    ("normal float32", 4, lambda n: tm.normal(KEY, n), lambda n: mx.random.normal((n,), key=MKEY), "normal", False),
    ("normal float32, odd start", 4, lambda n: tm.normal(KEY, n, position=32), None, "normal", False),
    ("exponential float32", 4, lambda n: tm.exponential(KEY, n), None, "uniform", True),
]


def rate(run, n, nbytes, calls=10, runs=5):
    run(n, 1)
    best = float("inf")
    for _ in range(runs):
        t0 = time.perf_counter()
        run(n, calls)
        best = min(best, time.perf_counter() - t0)
    return calls * n * nbytes / best / 2**30


print(f"{mx.device_info()['device_name']}, mlx {mx.__version__}")
print("| draw | log2 n | tandem_mlx | mx.random | MPS Philox |")
print("|---|---|---|---|---|")
for log2n in (24, 26):
    for label, nbytes, ours, theirs, kind, nearest in ROWS:
        n = 2**log2n
        cells = [f"{rate(mlx_calls(f), n, nbytes):.0f}" if f else "-" for f in (ours, theirs)]
        cells.append(f"{rate(philox(kind), n, nbytes):.0f}" + ("*" if nearest else ""))
        print(f"| {label} | {log2n} | " + " | ".join(cells) + " |")
