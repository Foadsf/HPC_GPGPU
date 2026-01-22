# OpenGL ES 2.0 GPGPU Matrix Multiplication

Demonstrates **actual GPU-accelerated** matrix multiplication on the Raspberry Pi 3B using VideoCore IV via OpenGL ES 2.0 fragment shader GPGPU.

## Performance Results

| Matrix | CPU Time | GPU Time | Speedup | GPU GFLOPS |
|--------|----------|----------|---------|------------|
| 64×64 | 18.89 ms | 1.23 ms | **15.3x** | 0.425 |
| 128×128 | 187.89 ms | 8.94 ms | **21.0x** | 0.469 |
| 256×256 | 1023.60 ms | 69.92 ms | **14.6x** | 0.480 |

## Build & Run
```bash
sudo apt install -y build-essential cmake libgles2-mesa-dev libegl1-mesa-dev libgbm-dev libdrm-dev
mkdir build && cd build
cmake .. && make
./gpgpu_mm [matrix_size] [iterations]
```

## License

MIT License
