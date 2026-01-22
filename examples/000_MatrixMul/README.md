# GPGPU Matrix Multiplication (OpenGL ES 2.0)

This example implements "Classic GPGPU" matrix multiplication on the Raspberry Pi 3 Model B.

## Architecture
Since the Broadcom VideoCore IV GPU does not support Compute Shaders (which require OpenGL ES 3.1+), this implementation uses the **Fragment Shader** technique (OpenGL ES 2.0):
1.  **Input:** Matrices $A$ and $B$ are encoded into RGBA textures.
2.  **Compute:** A fragment shader calculates the dot product for each pixel.
3.  **Output:** Results are rendered to a Framebuffer Object (FBO) and read back to the CPU.

## Implementation Details
* **Format:** `GL_RGBA` / `GL_UNSIGNED_BYTE` (Standard 8-bit channels).
* **Normalization:** The shader calculates the **Average** ($Sum / N$) rather than the Sum. This prevents the 8-bit color channels from saturating (clamping to 1.0) when values exceed 1.0.
* **Kernel:** The fragment shader iterates through the "row" of Texture A and "column" of Texture B to compute the result.

## Build & Run

### Prerequisites
* `libgles2-mesa-dev`
* `glfw3`
* `cmake`

### Compilation
```bash
mkdir build
cd build
cmake ..
cmake --build .

```

### Execution

To run on the hardware GPU, you need an active X session (desktop or VNC).

```bash
# If running from a desktop terminal or VNC:
./gpgpu_mm

# If running headless via SSH (requires Xvfb, runs in software emulation):
xvfb-run -a ./gpgpu_mm

```
