# 005_MultiCore_NEON_Intrinsics

High-performance matrix multiplication using **ARM NEON intrinsics** and **OpenMP** parallelization, targeting the **Raspberry Pi 3B** (Cortex-A53 quad-core).

## Overview

This example demonstrates how to maximize CPU throughput by combining:

| Optimization | Technique | Benefit |
|--------------|-----------|---------|
| **SIMD** | ARM NEON (128-bit vectors) | 4× parallelism per core |
| **Register Blocking** | 4×4 micro-kernel | Hides instruction latency |
| **Multi-threading** | OpenMP | 4× parallelism across cores |
| **Cache Optimization** | Matrix B transpose | Sequential memory access |

## Theoretical Performance Analysis

### Cortex-A53 Architecture (Pi 3B)

| Feature | Specification |
|---------|---------------|
| CPU Frequency | ~1.4 GHz |
| Cores | 4 (quad-core) |
| NEON Width | 128-bit (4 × float32) |
| Issue Width | Dual-issue (in-order) |
| L1 Data Cache | 32 KB per core |
| L2 Cache | 512 KB shared |

### Peak GFLOPS Calculation

```text
NEON FMA:     1 instruction × 4 floats × 2 FLOPs = 8 FLOP/cycle
Single Core:  1.4 GHz × 8 FLOP/cycle = 11.2 GFLOPS
Quad Core:    4 × 11.2 = 44.8 GFLOPS (theoretical maximum)

```

### Measured Performance vs. Reality

While the theoretical peak is **44.8 GFLOPS**, real-world performance is heavily constrained by the **Memory Wall** (RAM bandwidth) and thermal limits.

In our benchmarks (see below), we achieved **~3.72 GFLOPS**. While this is only ~8.3% of the theoretical peak, it represents a massive **270x speedup** over the naive implementation.

| Implementation | Measured Time (1024²) | Measured GFLOPS |
| --- | --- | --- |
| Naive (1 thread) | ~156 sec | ~0.01 |
| NEON (1 thread) | ~1.87 sec | ~1.15 |
| **NEON+OpenMP (4 threads)** | **~0.58 sec** | **~3.72** |

## Algorithm Details

### 4×4 Register Blocking

The key optimization is the **micro-kernel** that computes a 4×4 block of the output matrix:

```
C[i:i+4][j:j+4] = A[i:i+4][:] × B[:][j:j+4]

```

This produces **16 output values** using **4×4 = 16 FMAs** per iteration of the inner loop, which:

1. **Reuses loaded data**: Each row of A is used 4 times, each column of B is used 4 times.
2. **Hides latency**: 16 independent FMA operations overlap in the pipeline.
3. **Fits in registers**: Uses 12 of the 16 available Q registers.

### Register Allocation

```text
Registers:    Usage
────────────────────────────────────
q0-q3         4 rows of A (input)
q4-q7         4 columns of B^T (after transpose)
q8-q11        4 rows of C (accumulators)
q12-q15       Temporaries for micro-transpose

```

### OpenMP Parallelization

The outer loop over rows of C is parallelized:

```c
#pragma omp parallel for schedule(static)
for (int i = 0; i < n; i += 4) {
    // Each thread computes n/4/num_threads rows of C
    // Data remains in local L1 cache where possible
    for (int j = 0; j < n; j += 4) {
        kernel_4x4_neon(...);
    }
}

```

## Prerequisites

### Hardware

* Raspberry Pi 3B, 3B+, or 3A+ (Cortex-A53)
* **Note**: Pi 4/5 have different architectures (Cortex-A72/A76) and will perform significantly faster.

### Software

* Raspberry Pi OS (32-bit recommended for this assembly/intrinsic set)
* GCC with ARM NEON support
* OpenMP runtime
* CMake 3.10+

### Install Dependencies

```bash
sudo apt update
sudo apt install cmake build-essential

```

## File Structure

```
005_MultiCore_NEON_Intrinsics/
├── CMakeLists.txt          # Build configuration (NEON + OpenMP flags)
├── README.md               # This file
├── main.c                  # Benchmark driver
├── matmul_neon_omp.c       # NEON+OpenMP implementation
└── matmul_neon_omp.h       # Header file

```

## Building

```bash
mkdir build && cd build
cmake ..
make

```

### Verify Build Flags

The CMake output should show flags similar to:
`C Flags: -mcpu=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard -O3 -ffast-math -fopenmp`

## Running

```bash
./matmul_neon_omp

```

You can also specify a custom matrix size (default is 1024):

```bash
./matmul_neon_omp 512

```

## Actual Output (Raspberry Pi 3B)

```text
╔══════════════════════════════════════════════════════════════════════╗
║        005_MultiCore_NEON_Intrinsics - Matrix Multiplication         ║
║                     Raspberry Pi 3B (Cortex-A53)                     ║
╚══════════════════════════════════════════════════════════════════════╝

System Information:
  CPU:             Cortex-A53 @ 1.4 GHz (estimated)
  OpenMP Threads:  4
  SIMD:            ARM NEON (128-bit, 4×float)

Workload:
  Matrix Size:     1024 × 1024
  FLOPs:           2.15 GFLOP
  Memory:          12.00 MB (3 matrices)

Theoretical Peak Performance (Cortex-A53 @ 1.4 GHz):
  Single Core:     11.2 GFLOPS
  Quad Core:       44.8 GFLOPS
  Note: Memory bandwidth typically limits to 30-50% of peak.

Running benchmarks (1 warmup, 3 iterations each):

  [1/3] Naive triple-loop (single thread)...
        Done.
  [2/3] NEON intrinsics (single thread)...
        Done.
  [3/3] NEON intrinsics + OpenMP (4 threads)...
        Done.

══════════════════════════════════════════════════════════════════════════
Results (Matrix: 1024×1024, epsilon=1e-04):

  Naive (1 thread)                 156.205 sec      0.01 GFLOPS  [PASS]
  NEON (1 thread)                    1.868 sec      1.15 GFLOPS  [PASS]
  NEON+OpenMP (4 threads)            0.578 sec      3.72 GFLOPS  [PASS]

Speedup Analysis:
  NEON vs Naive:           83.61x
  NEON+OMP vs Naive:       270.43x
  NEON+OMP vs NEON:        3.23x (parallel efficiency: 81%)

Efficiency vs Theoretical Peak:
  NEON (1 thread):         10.3% of single-core peak (1.1 GFLOPS)
  NEON+OMP (4 threads):    8.3% of quad-core peak (3.7 GFLOPS)
══════════════════════════════════════════════════════════════════════════

```

## Performance Tuning Tips

1. **CPU Scaling**: Ensure the governor is set to performance to prevent clock down-throttling during the benchmark.
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

```


2. **Thread Pinning**: In some cases, explicitly binding threads to cores can reduce cache thrashing.
```bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores
./matmul_neon_omp

```


3. **Heat Management**: The Pi 3B will throttle quickly if all 4 cores are saturated. Ensure you have a heatsink or fan if running continuous benchmarks.

## References

* [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics)
* [Cortex-A53 Software Optimization Guide](https://developer.arm.com/documentation/uan0015/b/)
* [OpenMP Specification](https://www.openmp.org/specifications/)

## License

MIT License — See repository root for details.

```

```# 005_MultiCore_NEON_Intrinsics

This example demonstrates how to maximize the CPU's floating-point performance on the Raspberry Pi 3B by combining **NEON SIMD intrinsics** with **OpenMP multi-threading** and **cache tiling**.

## 🎯 Goal
To reach the "Speed of Light" for CPU-bound matrix multiplication ($C = A \times B$) on the Cortex-A53 before attempting GPU offloading. This serves as the baseline for judging GPU efficiency.

## 🏗 Architecture & Optimization

### 1. Multi-Threading (OpenMP)
The Raspberry Pi 3B has 4 Cortex-A53 cores. We use OpenMP to parallelize the outer loops, distributing work across all 4 cores.
* **Speedup:** ~3-4x over single-core.

### 2. SIMD Parallelism (NEON Intrinsics)
The Cortex-A53 has 128-bit NEON vector units. We use C-level intrinsics (`<arm_neon.h>`) to process 4 floats per instruction.
* **Key Intrinsic:** `vmlaq_lane_f32` (Vector Multiply-Accumulate Lane). This allows us to multiply a vector of $B$ by a scalar from $A$ and add to $C$ in a single cycle.
* **Speedup:** ~4x theoretical over scalar code.

### 3. Cache Tiling (The Critical Optimization)
Large matrices ($1024 \times 1024$) require 12MB of RAM, far exceeding the Pi 3B's **512KB L2 Cache**. A naive loop constantly thrashes the cache, fetching data from slow DRAM.
* **Solution:** We process the matrix in small **64x64 blocks (tiles)**.
* **Benefit:** A tile set fits entirely in the L2 cache, reusing loaded data hundreds of times before evicting it.

## 📊 Performance Results (Raspberry Pi 3B)

Benchmark run on $1024 \times 1024$ matrices:

| Implementation | Execution Time | Performance | Speedup | Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (Scalar)** | 156.21 s | 0.01 GFLOPS | 1x | Baseline |
| **NEON (1 Core)** | 1.87 s | 1.15 GFLOPS | **83x** | 10.3% of Peak |
| **NEON + OpenMP (4 Cores)** | **0.58 s** | **3.72 GFLOPS** | **270x** | 8.3% of Peak |

* **Parallel Efficiency:** 81% (Linear scaling would be 100%).
* **Theoretical Peak:** The Cortex-A53 @ 1.4GHz has a theoretical peak of ~44.8 GFLOPS (assuming perfect pipeline filling). We achieve **~8.3%** of this peak.
* **The Bottleneck:** Even with tiling, the Raspberry Pi 3B is heavily **memory bandwidth limited**. The single-channel LPDDR2 RAM cannot feed the 4 cores fast enough to saturate the ALUs fully.

## 🔨 Build Instructions

```bash
mkdir build
cd build
cmake ..
make

```

## 🚀 Running

```bash
# Run with root priority for more stable timing (optional)
sudo ./matmul_neon_omp

```

## 📂 Code Structure

* **`main.c`**: Orchestrates the benchmark, allocates memory, and verifies results against a naive reference.
* **`matmul_neon_omp.c`**: The optimized kernel.
* Uses `vld1q_f32` to load data.
* Uses `vmlaq_lane_f32` for computation.
* Implements 3-level loop tiling (i-tile, j-tile, k-tile).


* **`CMakeLists.txt`**: Configures GCC with `-O3`, `-mcpu=cortex-a53`, `-mfpu=neon-vfpv4`, and `-fopenmp`.

## 📚 References

* [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/intrinsics)
* [Cortex-A53 Processor Optimization Guide](https://developer.arm.com/documentation/uan0015/b/)
* [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
