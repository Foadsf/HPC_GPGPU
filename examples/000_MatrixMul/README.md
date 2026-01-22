# GPGPU Matrix Multiplication (OpenGL ES 2.0 / Half-Float)

This benchmark implements GPGPU Matrix Multiplication on the Raspberry Pi 3 Model B (VideoCore IV).

## Architecture
Since the VideoCore IV GPU lacks Compute Shaders (OpenGL 4.3), this implementation uses a **Fragment Shader** approach with **Half-Float Textures**.

* **Technique:** Render-to-Texture (Ping-Pong).
* **Precision:** 16-bit Floating Point (`GL_OES_texture_half_float`).
    * Unlike standard 8-bit textures (which clamp values to `1.0`), Half-Floats allow storing values up to ~65,504.
    * This enables actual general-purpose computation (e.g., matrices with values > 1.0).

## Hardware vs Software
* **Extension:** Requires `GL_OES_texture_half_float` and `GL_EXT_color_buffer_half_float`.
* **Verification:** Verified working on Raspberry Pi OS via `llvmpipe` (Software Rasterizer).
    * *Note on Hardware:* Native VideoCore IV support for rendering **to** half-float textures depends on the specific Mesa driver version active.

## Build & Run

### Prerequisites
* `libgles2-mesa-dev`
* `glfw3` (via vcpkg)
* `cmake`

### Compilation
```bash
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build .

```

### Execution

**Headless (Software Verification):**

```bash
xvfb-run -a ./gpgpu_mm

```

**Hardware (Requires Active X Session):**

```bash
./gpgpu_mm

```
