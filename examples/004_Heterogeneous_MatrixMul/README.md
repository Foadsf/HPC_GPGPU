# 004_Heterogeneous_MatrixMul

A bare-metal assembly implementation demonstrating **heterogeneous computing** with matrix/vector operations on both the ARM CPU (using NEON SIMD) and the VideoCore IV GPU (using QPU assembly).

## Overview

This MWE extends `003_Heterogeneous_Assembly` by performing actual computation rather than just a proof-of-concept memory write:

| Processor | Architecture | Operation | SIMD Width |
|-----------|--------------|-----------|------------|
| **ARM CPU** | ARMv7 NEON | 4×4 matrix multiply | 4 floats (128-bit) |
| **VideoCore IV GPU** | QPU | 16-element vector multiply | 16 floats (512-bit) |

## Test Results

```
=== 004_Heterogeneous_MatrixMul ===

[CPU] ARM NEON 4x4 Matrix Multiply
      Input A (row 0): [1.0, 2.0, 3.0, 4.0]
      Result C (row 0): [1.0, 2.0, 3.0, 4.0]
      Time: 2.00 µs
      Max error: 0.000000
      [PASS] NEON assembly verified

      Non-trivial test: A × B where A=B=[1..16]
      NEON C[0][0] = 90.0 (expected: 90.0)
      NEON C[3][3] = 600.0 (expected: 600.0)

[GPU] VideoCore IV QPU 16-Element Vector Multiply
      GPU memory allocated at bus addr: 0xFEA98000
      Copied QPU kernel (240 bytes)
      Vector A: [1, 2, 3, ..., 16]
      Vector B: [1, 2, 3, ..., 16]
      Expected C: [1, 4, 9, ..., 256]
      Executing QPU kernel...
      execute_qpu returned: 0x00000000
      Time: 106.00 µs
      Reading results...
      C[0]  = 1.0 (expected: 1.0)
      C[15] = 256.0 (expected: 256.0)
      Max error: 0.000000
      [PASS] QPU assembly verified

=== Performance Summary ===
      CPU (NEON 4×4 matmul): 2.00 µs
      GPU (QPU 16-vec mul):  106.00 µs
```

## Prerequisites

### Hardware
- Raspberry Pi 2, 3, or 3B+ (VideoCore IV GPU)
- **Not compatible with Pi 4/5** (different GPU architecture)

### Software
- Raspberry Pi OS (32-bit)
- GCC with NEON support
- CMake 3.10+
- **vc4asm** - VideoCore IV QPU assembler

### Critical: Disable the vc4 Graphics Driver

Direct QPU access requires disabling the kernel's vc4 driver:

```bash
sudo nano /boot/firmware/config.txt
# Comment out this line:
# dtoverlay=vc4-kms-v3d

sudo reboot
```

Verify it's disabled:
```bash
lsmod | grep vc4           # Should return nothing
ls -la /dev/dri/           # Should say "No such file or directory"
```

> ⚠️ **Warning**: Disabling vc4 driver degrades desktop graphics. Re-enable when needed.

## File Structure

```
004_Heterogeneous_MatrixMul/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── main.c                  # Host orchestrator
├── cpu_matmul_neon.s       # ARM NEON 4×4 matrix multiply
└── qpu_matmul.qasm         # QPU 16-element vector multiply
```

## Building

```bash
mkdir build && cd build
cmake ..
make
```

## Running

```bash
sudo ./matmul_asm
```

## How It Works

### ARM NEON Assembly (`cpu_matmul_neon.s`)

Performs 4×4 single-precision matrix multiplication using NEON SIMD:

```asm
@ Load 4 floats (one row of A) into 128-bit register
vld1.32 {q0}, [r0]!

@ Transpose B matrix for efficient column access
vtrn.32 q4, q5
vswp    d9, d12

@ Element-wise multiply
vmul.f32 q8, q0, q4

@ Horizontal add for dot product reduction
vpadd.f32 d16, d16, d17
```

**Key NEON concepts:**
- `q0-q15`: 128-bit registers holding 4 floats each
- `vld1.32`: Load 4 consecutive floats
- `vtrn.32`: Transpose 2×2 blocks (for matrix transpose)
- `vpadd.f32`: Pairwise add (horizontal reduction)

### QPU Assembly (`qpu_matmul.qasm`)

Performs 16-element vector multiply using VideoCore IV QPU:

```asm
# Each of 16 SIMD lanes computes its own address
mov r1, elem_num            # r1 = lane number [0..15]
shl r2, r1, 2               # r2 = lane * 4 (byte offset)
add r2, ra_addr_a, r2       # r2 = &A[lane]

# Request memory fetch via TMU
mov tmu0_s, r2

# ... wait for TMU latency ...

# Receive result (TMU always returns to r4)
nop; ldtmu0
mov ra_a_vec, r4            # All 16 elements loaded

# Parallel multiply (16 floats at once)
fmul r0, ra_a_vec, rb_b_vec

# Write result via VPM + DMA
ldi vw_setup, 0x00001a00    # Configure VPM
mov vpm, r0                  # Write to VPM
ldi vw_setup, 0x80904000    # Configure DMA
mov vw_addr, ra_addr_out    # Trigger DMA to memory
```

**Key QPU concepts:**

| Concept | Description |
|---------|-------------|
| **TMU** | Texture Memory Unit - fetches data from main memory |
| **VPM** | Vertex Pipe Memory - 4KB scratchpad for DMA transfers |
| **VDW** | VPM DMA Writer - transfers VPM data to main memory |
| **elem_num** | Built-in register giving SIMD lane number (0-15) |
| **r4** | Special register that receives TMU results |
| **Uniforms** | Parameters passed from host CPU to QPU |

### Register File Constraints

The VideoCore IV has a critical constraint: **in one instruction, you can read at most one register from file A and one from file B**.

```asm
# ILLEGAL - two reads from file A:
add r0, ra0, ra1            # ERROR!

# LEGAL - one from A, one from B:
add r0, ra0, rb0            # OK

# LEGAL - accumulators have no restriction:
add r0, r1, r2              # OK (r0-r3 are accumulators)
```

### Memory Layout

```
GPU Memory (64KB allocation):
┌─────────────────────────────────────┐ 0x0000
│ QPU Code (kernel binary)            │
├─────────────────────────────────────┤ 0x1000
│ Uniforms                            │
│   [0] = address of A                │
│   [1] = address of B                │
│   [2] = address of C                │
│   [3] = element count               │
├─────────────────────────────────────┤ 0x2000
│ Vector A (16 floats = 64 bytes)     │
├─────────────────────────────────────┤ 0x2100
│ Vector B (16 floats = 64 bytes)     │
├─────────────────────────────────────┤ 0x2200
│ Vector C output (16 floats)         │
├─────────────────────────────────────┤ 0x3000
│ Control List                        │
│   [0] = uniforms address            │
│   [1] = code address                │
└─────────────────────────────────────┘
```

## Architecture Comparison

| Feature | ARM NEON | VideoCore IV QPU |
|---------|----------|------------------|
| SIMD Width | 4 floats (128-bit) | 16 floats (512-bit) |
| Registers | 32 × 128-bit (q0-q15, aliased) | 2 × 32 × 32-bit (ra/rb files) |
| Memory Access | Direct load/store | TMU (read), VPM+DMA (write) |
| Peak GFLOPS | ~4 GFLOPS | ~24 GFLOPS (12 QPUs) |
| Latency | Low | High (TMU: ~9 cycles) |

## Performance Notes

The GPU timing (106 µs) includes significant overhead:
- Mailbox communication with VideoCore firmware
- Memory allocation and mapping
- DMA transfer setup and completion

For production workloads, this overhead is amortized over:
- Larger data sizes (thousands of elements)
- Multiple iterations
- Multi-QPU parallelism (using all 12 QPUs)

See [gpu_fft](http://www.aholme.co.uk/GPU_FFT/Main.htm) and [pi-gemm](https://github.com/jetpacapp/pi-gemm) for optimized implementations that achieve near-peak performance.

## Troubleshooting

### "Cannot open /dev/vcio"
- Run with `sudo`
- Check that `/dev/vcio` exists

### "GPU memory allocation failed"
- Ensure vc4 driver is disabled (`lsmod | grep vc4` should be empty)
- Check GPU memory: `vcgencmd get_mem gpu` (should be ≥64MB)
- Add `gpu_mem=128` to `/boot/firmware/config.txt`

### QPU results are all zeros
- Verify vc4 driver is disabled
- Check that `execute_qpu` doesn't return an error
- Ensure memory addresses are bus addresses (not physical)

### NEON assembly errors
- Ensure 32-bit Raspberry Pi OS (not 64-bit)
- Check that NEON is enabled: `cat /proc/cpuinfo | grep neon`

## References

- [VideoCore IV 3D Architecture Guide](https://docs.broadcom.com/doc/12358545) (Broadcom)
- [vc4asm Documentation](https://maazl.de/project/vc4asm/doc/index.html)
- [vc4asm Addendum](https://maazl.de/project/vc4asm/doc/VideoCoreIV-addendum.html) - Important errata
- [Pete Warden: Optimizing RPi GPU Code](https://petewarden.com/2014/08/07/how-to-optimize-raspberry-pi-code-using-its-gpu/)
- [ARM NEON Intrinsics Reference](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon)
- [gpu-deadbeef VPM Tutorial](https://github.com/0xfaded/gpu-deadbeef) - Excellent VPM explanation

## License

MIT License — See repository root for details.
