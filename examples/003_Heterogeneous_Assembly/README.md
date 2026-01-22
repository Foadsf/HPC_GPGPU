# 003_Heterogeneous_Assembly

A minimal working example demonstrating **bare-metal assembly execution** on both the ARM CPU and VideoCore IV GPU of the Raspberry Pi.

## Overview

This MWE proves that you can write and execute raw assembly code on both processors in a Raspberry Pi:

| Processor | Architecture | Assembly File | What It Does |
|-----------|--------------|---------------|--------------|
| **ARM CPU** | ARMv7 (32-bit) | `cpu_ops.s` | Adds two integers |
| **VideoCore IV GPU** | QPU (12-way SIMD) | `qpu_kernel.qasm` | Writes `0x1337` to shared memory via VPM/DMA |

## Prerequisites

### Hardware
- Raspberry Pi 2, 3, or 3B+ (VideoCore IV GPU)
- **Not compatible with Pi 4/5** (different GPU architecture)

### Software
- Raspberry Pi OS (32-bit recommended)
- GCC toolchain (`build-essential`)
- CMake 3.10+
- **vc4asm** - VideoCore IV QPU assembler

### Critical: Disable the vc4 Graphics Driver

The vc4-kms-v3d driver monopolizes the V3D hardware. For bare-metal QPU access, you must disable it:

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

> ⚠️ **Warning**: Disabling the vc4 driver degrades desktop graphics performance. Re-enable it when you need normal GUI operation.

## Installing vc4asm

```bash
# Install dependencies
sudo apt update
sudo apt install -y git cmake build-essential

# Clone and build
cd ~
git clone https://github.com/maazl/vc4asm.git
cd vc4asm
mkdir build && cd build
cmake ..
make
sudo make install

# Verify
vc4asm --version
```

## Building

```bash
cd ~/HPC_GPGPU/examples/003_Heterogeneous_Assembly
mkdir build && cd build
cmake ..
make
```

## Running

```bash
sudo ./rpi_asm_host
```

### Expected Output

```
--- 003_Heterogeneous_Assembly ---

[CPU] Testing ARMv7 Assembly...
      10 + 20 = 30 (Calculated by cpu_ops.s)
      [PASS]

[GPU] Preparing VideoCore IV QPU...
      Mailbox opened successfully.
      QPU enabled.
      Allocated GPU memory, handle: 0x00000005
      Locked memory at bus address: 0xFEAA7000
      Physical address: 0x3EAA7000
      Mapped GPU memory to user space.
      Copying QPU kernel (88 bytes)...
      Uniforms set. Output address: 0xFEAA7200
      Initial output value: 0xDEADBEEF

[GPU] Launching QPU Kernel...
      execute_qpu returned: 0x00000000

[GPU] Verifying Results...
      Read back value: 0x00001337
      [SUCCESS] QPU wrote the magic number!

[CLEANUP] Releasing resources...
      Done.
```

## File Descriptions

### `cpu_ops.s` — ARM CPU Assembly

Simple ARMv7 assembly function that adds two integers:

```asm
.global cpu_add_asm
cpu_add_asm:
    add r0, r0, r1    @ r0 = r0 + r1
    bx  lr            @ Return
```

- Uses ARM calling convention: arguments in `r0`, `r1`; return value in `r0`
- Called from C as `int cpu_add_asm(int a, int b)`

### `qpu_kernel.qasm` — VideoCore IV QPU Assembly

QPU assembly that writes a magic number to shared memory:

```asm
mov r0, unif              # Load memory address from uniform
ldi r1, 0x1337            # Load magic value
ldi vw_setup, 0x00001a00  # Configure VPM for 32-bit write
mov vpm, r1               # Write to VPM
ldi vw_setup, 0x80904000  # Configure DMA store
mov vw_addr, r0           # Trigger DMA to memory address
mov r2, vw_wait           # Wait for DMA completion
mov interrupt, 1          # Signal host
thrend                    # End thread
```

Key concepts:
- **VPM (Vertex Pipe Memory)**: QPUs cannot write directly to RAM; they must use VPM as an intermediary
- **DMA**: The VPM DMA engine transfers data between VPM and system RAM
- **Uniforms**: Parameters passed from the host (CPU) to the QPU

### `main.c` — Host Orchestrator

The C program that:
1. Calls the ARM assembly function to prove CPU-side assembly works
2. Opens the VideoCore mailbox interface (`/dev/vcio`)
3. Allocates GPU-accessible memory via mailbox
4. Copies the QPU kernel to GPU memory
5. Sets up uniforms (parameters)
6. Executes the QPU kernel via `execute_qpu` mailbox call
7. Reads back and verifies the result

### `CMakeLists.txt` — Build System

Handles:
- Finding `vc4asm` executable
- Custom command to assemble `.qasm` → `.hex`
- Linking ARM assembly with C code

## Architecture Details

### Memory Layout

```
GPU Memory (4KB allocation):
┌────────────────────────────────────────┐
│ Offset 0:    QPU Code (88 bytes)       │
├────────────────────────────────────────┤
│ Offset 256:  Uniforms (control data)   │
│              [0] = output bus address  │
├────────────────────────────────────────┤
│ Offset 512:  Output Data               │
│              [0] = result (0x1337)     │
├────────────────────────────────────────┤
│ Offset 768:  Control List              │
│              [0] = uniforms address    │
│              [1] = code address        │
└────────────────────────────────────────┘
```

### Mailbox Interface

Communication with the VideoCore firmware uses the mailbox property interface:

| Tag | Function |
|-----|----------|
| `0x3000c` | Allocate GPU memory |
| `0x3000d` | Lock memory (get bus address) |
| `0x3000e` | Unlock memory |
| `0x3000f` | Free memory |
| `0x30011` | Execute QPU code |
| `0x30012` | Enable/disable QPU |

### Bus vs Physical Addresses

- **Bus address**: What the GPU sees (e.g., `0xFEAA7000`)
- **Physical address**: What the CPU's MMU sees (`bus_addr & ~0xC0000000`)
- The `0xC0000000` mask converts between GPU's L2-cached alias and direct physical memory

## Troubleshooting

### "GPU Alloc failed"
- Check GPU memory: `vcgencmd get_mem gpu` (should be ≥64MB)
- Ensure vc4 driver is disabled
- Try: `gpu_mem=128` in `/boot/firmware/config.txt`

### "ioctl mbox_property failed: Connection timed out"
- The vc4-kms-v3d driver is still loaded
- Verify: `lsmod | grep vc4` should return nothing
- Reboot after modifying config.txt

### "execute_qpu returned: 0x80000000"
- QPU execution failed (timeout or error)
- Check kernel code for assembly errors
- Verify memory alignment (4KB for code)

### Linker warning about `.note.GNU-stack`
- Harmless warning; add to `cpu_ops.s`:
  ```asm
  .section .note.GNU-stack,"",%progbits
  ```

## References

- [VideoCore IV 3D Architecture Guide](https://docs.broadcom.com/doc/12358545) (Broadcom)
- [vc4asm Documentation](https://maazl.de/project/vc4asm/doc/index.html)
- [Raspberry Pi Mailbox Property Interface](https://github.com/raspberrypi/firmware/wiki/Mailbox-property-interface)
- [GPU_FFT by Andrew Holme](http://www.aholme.co.uk/GPU_FFT/Main.htm) — canonical QPU programming reference

## License

MIT License — See repository root for details.

## Author

Part of the [HPC_GPGPU](https://github.com/Foadsf/HPC_GPGPU) examples collection.
