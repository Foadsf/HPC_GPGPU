# Matrix Multiplication: GPGPU (Software) vs Native CPU

A benchmark comparing **Single-Threaded CPU** performance against **Software-Accelerated OpenGL Compute Shaders** (via llvmpipe) on the Raspberry Pi 3 Model B v1.2.

## The Benchmark
* **Task:** Calculate $C = A \times B$ for $1024 \times 1024$ matrices.
* **Complexity:** Requires approx. 2 Billion floating-point operations.
* **Hardware:** Raspberry Pi 3 Model B v1.2 (Broadcom BCM2837, Cortex-A53 @ 1.2GHz).

## Performance Results

| Implementation | Execution Path | Execution Time | Speedup |
| :--- | :--- | :--- | :--- |
| **Native CPU** | Single-threaded C++ Loop | ~208,104 ms (3.5 min) | 1x |
| **OpenGL Compute** | `llvmpipe` (Multi-threaded CPU SIMD) | ~17,718 ms (17.7 sec) | **11.7x** |

### Comparison Analysis
Even though the Raspberry Pi 3B's GPU (VideoCore IV) does not natively support Compute Shaders, using the **OpenGL software rasterizer (llvmpipe)** provided an **11x speedup** over the naive C++ implementation.

* **Why?** The native C++ implementation uses a naive triple-loop running on a single core. The `llvmpipe` driver automatically utilizes **SIMD (NEON) instructions** and **multi-threading** across all 4 cores of the Cortex-A53 to execute the shader math efficiently.

## Build Instructions

### Prerequisites
* Raspberry Pi 3B running Raspberry Pi OS (Bookworm)
* `vcpkg` for dependency management
* `xvfb` for headless execution

### Build
```bash
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build

```

### Run (Headless)

Since the Pi 3B requires software rendering for OpenGL 4.3:

```bash
xvfb-run -a env LIBGL_ALWAYS_SOFTWARE=1 ./build/gpgpu_mm

```
