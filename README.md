# HPC & GPGPU on Raspberry Pi 3B

This repository explores High-Performance Computing (HPC) and General-Purpose GPU (GPGPU) techniques on the **Raspberry Pi 3 Model B v1.2** (Broadcom BCM2837).

The goal is to extract parallel performance from legacy hardware that lacks modern GPGPU compute primitives (like OpenCL 1.2+ or OpenGL 4.3+).

## Projects

### [000_MatrixMul](examples/000_MatrixMul)
A $1024 \times 1024$ Matrix Multiplication benchmark using **OpenGL ES 2.0**.
* **Technique:** Render-to-Texture (Fragment Shader).
* **Precision:** **Half-Float (FP16)** via `GL_OES_texture_half_float`.
* **Hardware Support:** Verified on VideoCore IV. Uses texture attachments to perform parallel dot products, bypassing the lack of Compute Shaders.

## Requirements
* **OS:** Raspberry Pi OS (Bookworm)
* **Libraries:** `libgles2-mesa-dev`, `glfw3` (via vcpkg)
* **Tools:** `cmake`, `g++`, `git`

## License
MIT License.
