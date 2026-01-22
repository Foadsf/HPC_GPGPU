# VC4CL OpenCL Matrix Multiplication (Raspberry Pi 3B)

This project demonstrates General-Purpose GPU (GPGPU) programming on the Raspberry Pi 3B using the VideoCore IV GPU. It implements Matrix Multiplication ($C = A \times B$) using OpenCL and compares the performance against a scalar CPU implementation.

## 🎯 Project Goal
To benchmark and optimize compute performance on the Raspberry Pi's GPU using the **VC4CL** OpenCL implementation.

## 🛠 Prerequisites

* **Hardware:** Raspberry Pi 3B (VideoCore IV GPU)
* **OS:** Raspberry Pi OS (Legacy/Buster recommended for VC4CL stability)
* **Drivers:** [VC4CL](https://github.com/doe300/VC4CL) OpenCL implementation
* **Build Tools:** `cmake`, `make`, `g++`

## 🚀 Build Instructions

1.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```
2.  Configure with CMake:
    ```bash
    cmake ..
    ```
3.  Compile:
    ```bash
    make
    ```

## ⚡ Usage

**Note:** The VC4CL driver requires root privileges to access GPU memory (via `/dev/mem`).

```bash
sudo ./vc4cl_mm [Matrix_Size] [Iterations]

```

* `Matrix_Size`: Dimension of the square matrix (N x N). Range: 8 to 1024.
* `Iterations`: Number of times to run the benchmark for averaging.

### Example

```bash
# Run a 512x512 matrix multiplication for 10 iterations
sudo ./vc4cl_mm 512 10

```

## 📊 Performance Analysis

During development, we observed three distinct performance tiers:

| Implementation | Matrix Size | Performance (GFLOPS) | Speedup vs CPU | Notes |
| --- | --- | --- | --- | --- |
| **CPU (Scalar)** | 64 - 256 | ~0.05 GFLOPS | 1.0x | Reference baseline. |
| **GPU (Scalar)** | Any | ~0.02 GFLOPS | ~0.4x | Slower than CPU. High overhead, low lane utilization. |
| **GPU (Vector)** | 512+ | **~0.31 GFLOPS** | **~25x - 30x** | Uses `float16` vector types. Limited by memory bandwidth. |

### The "Memory Wall"

The VideoCore IV GPU is capable of ~24 GFLOPS theoretically. However, at **~0.31 GFLOPS**, performance plateaus. This is due to the shared memory architecture of the Raspberry Pi. The QPUs (compute units) consume data faster than the system RAM can provide it.

## 📂 File Structure

* `main.cpp`: Host code. Sets up OpenCL, generates random data, benchmarks CPU vs GPU, and verifies results.
* *Modifications:* Kernel build flags set to `-cl-fast-relaxed-math` for performance.
* *Modifications:* Thread count adjusted for 1:16 vectorization.


* `matmul.cl`: The OpenCL kernel code running on the GPU.
* *Current State:* **Vectorized**. Uses `float16` to compute 16 elements per thread.


* `CMakeLists.txt`: Build configuration linking against `libOpenCL`.

## ⚠️ Troubleshooting

**1. `Failed to open /dev/mem: Permission denied**`

* **Fix:** You must run the executable with `sudo`. VideoCore IV requires direct memory access.

**2. System Freeze / Timeout**

* **Cause:** The GPU is non-preemptive. If a kernel takes too long (>2 seconds), the display may freeze or the driver may reset.
* **Fix:** Keep matrix sizes ≤ 1024.

**3. Accuracy Errors**

* **Observation:** You may see "Errors > 0.001" in the summary.
* **Reason:** We use `-cl-fast-relaxed-math`. The GPU sacrifices strict IEEE 754 precision for speed, which is standard behavior for high-performance graphics/compute tasks.

---

*Created as part of the HPC GPGPU Learning Series.*
