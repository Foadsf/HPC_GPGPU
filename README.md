# HPC & GPGPU on Raspberry Pi 3B

This repository explores High-Performance Computing (HPC) and General-Purpose GPU (GPGPU) techniques on the **Raspberry Pi 3 Model B v1.2** (Broadcom BCM2837).

The goal is to extract parallel performance from legacy hardware that lacks modern GPGPU compute primitives.

## Projects

### [000_MatrixMul](examples/000_MatrixMul)
A $1024 \times 1024$ Matrix Multiplication benchmark using **OpenGL ES 2.0**.
* **Method:** Render-to-Texture (Fragment Shader).
* **Target:** VideoCore IV GPU.
* **Performance:** Demonstrates massively parallel arithmetic using the graphics pipeline.

## Requirements
* **OS:** Raspberry Pi OS (Bookworm)
* **Libraries:** `libgles2-mesa-dev`, `glfw3`
* **Tools:** `cmake`, `g++`, `git`

## License
MIT License.
