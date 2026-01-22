# HPC & GPGPU on Raspberry Pi 3B

This repository explores High-Performance Computing (HPC) and General-Purpose GPU (GPGPU) concepts on the **Raspberry Pi 3 Model B v1.2**.

It demonstrates how to leverage parallel computing techniques—even on hardware that lacks modern GPGPU drivers—by utilizing software-accelerated drivers (`llvmpipe`), SIMD instructions, and multi-threading.

## Repository Structure

* **`examples/000_MatrixMul`**: A $1024 \times 1024$ Matrix Multiplication benchmark.
    * Compares naive Single-Threaded CPU execution vs. Optimized Software Rendering (OpenGL via `llvmpipe`).
    * **Result:** 11.7x speedup achieved using `llvmpipe` on the Cortex-A53 CPU compared to a raw single-threaded C++ loop.

## Prerequisites

* **Hardware:** Raspberry Pi 3 Model B v1.2
* **OS:** Raspberry Pi OS (Debian Bookworm)
* **Tools:** `cmake`, `g++`, `vcpkg`, `git`, `gh`

## Setup

1. **Install Dependencies:**
   ```bash
   sudo apt install cmake g++ xvfb pkg-config libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libgl-dev -y

```

2. **Setup vcpkg:**
Ensure `vcpkg` is installed in your home directory and `glad` + `glfw3` are installed.
```bash
~/vcpkg/vcpkg install glad glfw3

```



## License

MIT License. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.
