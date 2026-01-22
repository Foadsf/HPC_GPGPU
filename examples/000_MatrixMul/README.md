# OpenGL ES 2.0 GPGPU Matrix Multiplication

A benchmark demonstrating **actual GPU-accelerated** matrix multiplication on the Raspberry Pi 3 Model B v1.2 using the VideoCore IV GPU via OpenGL ES 2.0 fragment shader GPGPU techniques.

## Performance Results

| Implementation | Execution Path | Execution Time |
| :--- | :--- | :--- |
| **Native CPU** | Single-threaded C++ loop | ~19 ms |
| **OpenGL ES 2.0 GPU** | VideoCore IV fragment shader | ~28 ms |

## Build Instructions
```bash
sudo apt install -y build-essential cmake libgles2-mesa-dev libegl1-mesa-dev libgbm-dev libdrm-dev
mkdir build && cd build
cmake .. && make
./gpgpu_mm
```

## License

MIT License
