# 006_Zero_Copy_Shared_Memory

**Micro-benchmark demonstrating the bandwidth benefits of Zero-Copy architecture on the Raspberry Pi 3B's Unified Memory System.**

This example is a critical lesson in understanding the **Memory Wall** and why GPU computing on systems with unified memory (like the Raspberry Pi) can benefit from avoiding unnecessary data copies.

## 🎯 Goal

Prove through measurement that:

1. **Eliminating the copy step** results in significantly faster data transfer to GPU-accessible memory.
2. **Direct/Uncached writes** can sometimes outperform cached writes for large datasets by avoiding cache pollution.
3. Understanding **bus address aliases** is essential for correct GPU memory access.

## 📚 Theoretical Background

### The Memory Wall

The "Memory Wall" refers to the growing disparity between CPU speed and memory bandwidth. On the Raspberry Pi 3B:

| Component | Speed/Bandwidth |
|-----------|-----------------|
| CPU Frequency | 1.4 GHz |
| L1 Cache | ~32 GB/s (32 KB) |
| L2 Cache | ~8 GB/s (512 KB) |
| **LPDDR2 RAM** | **~3.2 GB/s** (theoretical peak) |

The CPU can execute billions of operations per second, but **memory bandwidth limits actual throughput** to ~3 GB/s for sequential access.

### Unified Memory Architecture

Unlike desktop GPUs with dedicated VRAM connected via PCIe, the Raspberry Pi's VideoCore IV GPU shares the same physical RAM as the ARM CPU:


```

┌─────────────────────────────────────────────────────────────────┐
│                       LPDDR2 RAM (1 GB)                         │
│  ┌─────────────────────────────┬───────────────────────────┐    │
│  │        ARM Memory           │        GPU Memory         │    │
│  │      (Managed by Linux)     │    (Managed by Firmware)  │    │
│  │                             │                           │    │
│  │  ┌───────────┐              │    ┌──────────────────┐   │    │
│  │  │ User      │              │    │ GPU Allocated    │   │    │
│  │  │ Buffers   │  ──copy──►   │    │ Memory           │   │    │
│  │  └───────────┘              │    └──────────────────┘   │    │
│  │        │                    │             ▲             │    │
│  │        └────────────────────┼─────────────┘             │    │
│  │             Zero-Copy Path  │   (mmap to same physical) │    │
│  └─────────────────────────────┴───────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

```

**Key Insight**: Since the memory is shared, we can potentially avoid copying by having the CPU write directly to GPU-accessible addresses.

### Bus Addresses vs Physical Addresses

The BCM2837 uses **bus addresses** with alias bits to control caching behavior:


```

Bus Address Format:
┌─────────┬──────────────────────────────────────────┐
│ 31..30  │  29..0                                   │
│ (Alias) │  (Physical Address)                      │
└─────────┴──────────────────────────────────────────┘

```

| Alias | Bus Base | Caching Behavior | Use Case |
|-------|----------|------------------|----------|
| **0x0** | 0x00000000 | L1 + L2 Cached | Normal ARM access |
| **0x4** | 0x40000000 | L2 Cached, Coherent | Shared CPU/GPU data |
| **0x8** | 0x80000000 | L2 Cached (allocating) | GPU texture data |
| **0xC** | 0xC0000000 | **Direct/Uncached** | DMA, Zero-Copy |

When we allocate memory with `MEM_FLAG_DIRECT`, the firmware returns a bus address in the `0xC` range, ensuring the CPU bypasses caches when accessing it.

## 🔬 Benchmark Design

### Approach 1: Standard Copy


```

┌──────────────┐    fill     ┌──────────────┐    memcpy   ┌──────────────┐
│   malloc()   │ ─────────► │ CPU Buffer   │ ─────────► │ GPU Buffer   │
│   (cached)   │  (fast)    │ (L1/L2 hits) │  (slow)    │ (coherent)   │
└──────────────┘            └──────────────┘            └──────────────┘
│                               │                           │
│◄──── Cache friendly ─────►│◄─── Memory bandwidth ────►│

Total Time = Fill Time + Copy Time

```

**Characteristics**:
- Fill is typically fast (CPU writes to cached memory)
- Copy is slow (reads cached, writes to coherent memory)
- Double memory traffic (read + write)

### Approach 2: Zero-Copy Direct


```

┌──────────────┐    direct fill    ┌──────────────┐
│  mem_alloc() │ ──────────────► │ GPU Buffer   │
│  (uncached)  │  (write comb)   │ (0xC alias)  │
└──────────────┘                 └──────────────┘
│                                    │
│◄────── Every write hits RAM ──►│

Total Time = Direct Write Time (no copy!)

```

**Characteristics**:
- Each write goes directly to RAM
- Bypasses L1/L2 cache (no eviction overhead)
- **But no copy step!**

## 🔧 Implementation Details

### Mailbox Interface

The `mailbox.c/h` library provides a clean API for GPU memory management:

```c
// Open mailbox interface
mbox_handle_t mbox = mbox_open();

// Allocate GPU memory (returns handle)
uint32_t handle = mem_alloc(mbox, size, alignment, flags);

// Lock to get bus address
uint32_t bus_addr = mem_lock(mbox, handle);

// Map to user space
void *ptr = mapmem(bus_to_phys(bus_addr), size);

// ... use the memory ...

// Cleanup
unmapmem(ptr, size);
mem_unlock(mbox, handle);
mem_free(mbox, handle);
mbox_close(mbox);

```

### Memory Flags

| Flag | Value | Effect |
| --- | --- | --- |
| `MEM_FLAG_DIRECT` | 0x04 | Uncached (0xC alias) |
| `MEM_FLAG_COHERENT` | 0x08 | L2 coherent (0x4 alias) |
| `MEM_FLAG_ZERO` | 0x10 | Zero-initialize |
| `MEM_FLAG_ZERO_COPY` | 0x14 | DIRECT + ZERO |

## 📂 File Structure

```
006_Zero_Copy_Shared_Memory/
├── CMakeLists.txt      # Build configuration
├── README.md           # This documentation
├── mailbox.h           # Reusable mailbox library header
├── mailbox.c           # Mailbox implementation
└── main.c              # Benchmark driver

```

## 🔨 Building

```bash
mkdir build && cd build
cmake ..
make

```

## 🚀 Running

**Root privileges required** (for `/dev/vcio` and `/dev/mem` access):

```bash
# Default: 64 MB data
sudo ./zero_copy_bench

# Custom size (MB)
sudo ./zero_copy_bench 32    # 32 MB
sudo ./zero_copy_bench 128   # 128 MB

```

## 📈 Actual Output (Raspberry Pi 3B)

```
╔══════════════════════════════════════════════════════════════════════╗
║          006_Zero_Copy_Shared_Memory - Bandwidth Benchmark           ║
║                      Raspberry Pi 3B (BCM2837)                       ║
╚══════════════════════════════════════════════════════════════════════╝

Configuration:
  Data Size:   64 MB (67108864 bytes)
  Warmup:      1 iterations
  Iterations:  5 (averaged)

System Information:
  Firmware:    0x68A60009
  ARM Memory:  0x00000000 - 0x37FFFFFF (896 MB)
  GPU Memory:  0x38000000 - 0x3FFFFFFF (128 MB)

Memory Address Aliases (BCM2837):
  ┌─────────┬──────────────┬─────────────────────────────┐
  │ Alias   │ Base Address │ Caching                     │
  ├─────────┼──────────────┼─────────────────────────────┤
  │ 0x0     │ 0x00000000   │ L1 & L2 cached              │
  │ 0x4     │ 0x40000000   │ L2 coherent (ARM visible)   │
  │ 0x8     │ 0x80000000   │ L2 cached (allocating)      │
  │ 0xC     │ 0xC0000000   │ Direct/Uncached (bypass)    │
  └─────────┴──────────────┴─────────────────────────────┘

══════════════════════════════════════════════════════════════════════════
Running Benchmarks...

  Running Baseline (cached malloc) benchmark...

  Running Standard Copy benchmark...
    CPU buffer: 0x72dbc000 (cached)
    GPU buffer: 0x6edbb000 (bus: 0xBAAA8000, coherent)

  Running Zero-Copy Direct benchmark...
    GPU buffer: 0x72dbd000 (bus: 0xFAAA8000, direct/uncached)
    Alias: 0x3 (expected: 0xC for direct)

══════════════════════════════════════════════════════════════════════════
Results Summary (64 MB data):

  ┌────────────────────────────┬─────────────┬─────────────┬──────────┐
  │ Benchmark                  │ Time (ms)   │ BW (GB/s)   │ Status   │
  ├────────────────────────────┼─────────────┼─────────────┼──────────┤
  │ Baseline (cached malloc)   │      50.16  │     1.246   │  REF     │
  ├────────────────────────────┼─────────────┼─────────────┼──────────┤
  │ Standard: Fill (cached)    │      45.51  │     1.373   │          │
  │ Standard: Copy (memcpy)    │      64.04  │     0.976   │          │
  │ Standard: TOTAL            │     109.56  │     0.570   │ [PASS]   │
  ├────────────────────────────┼─────────────┼─────────────┼──────────┤
  │ Zero-Copy: Direct write    │      38.26  │     1.633   │ [PASS]   │
  └────────────────────────────┴─────────────┴─────────────┴──────────┘

Analysis:
  ✓ Zero-Copy is 2.86x FASTER than Standard approach
  • Copy overhead in Standard: 58.5% of total time
  • Uncached write bonus: 1.31x faster than cached baseline!

Key Insights:
  1. Surprisingly, uncached writes (1.63 GB/s) >= cached (1.25 GB/s)!
  2. This suggests write-combining or store buffers are effective
  3. Standard STILL pays copy overhead (58.5% of time)
  4. For 64 MB transfers, eliminating the copy wins!

```

## 🧠 Understanding the Results

### The Surprise: Why Uncached Writes were Faster?

Contrary to basic theory, our benchmark showed **Direct Writes (1.63 GB/s)** outperforming **Cached Writes (1.25 GB/s)**.

This usually happens with large, sequential datasets (like our 64MB buffer) because of:

1. **Write Combining**: The ARM CPU has store buffers that merge consecutive small writes into single large burst transactions to RAM. Since our data pattern is perfectly sequential, this mechanism is highly effective.
2. **No Cache Thrashing**: Writing 64MB of data to a CPU with only 512KB of L2 cache causes massive "cache pollution." Old lines are constantly evicted to make room for new ones. Direct writes bypass the cache entirely, avoiding this eviction overhead.

### The Copy Tax

The Standard approach paid a heavy price:

1. **Fill**: 45.51 ms (writing to cache/RAM)
2. **Copy**: 64.04 ms (reading back from RAM, writing to new RAM location)
3. **Total**: ~110 ms

The Zero-Copy approach only paid once:

1. **Direct**: 38.26 ms
2. **Total**: ~38 ms

### Conclusion

For high-bandwidth applications on the Raspberry Pi 3B (like feeding the QPU or video processing), **Zero-Copy is mandatory**. It offers nearly **3x the performance** of the standard `malloc` + `memcpy` approach for large transfers.

## 🔗 Connection to GPU Computing

This benchmark sets the stage for the next examples:

1. **QPU Code Execution**: QPUs read data from bus addresses. Using `MEM_FLAG_DIRECT` ensures data is in RAM when the QPU starts.
2. **Streaming Pipelines**: For real-time processing, Zero-Copy allows continuous data flow without synchronization delays.

## 📚 References

* [BCM2835 ARM Peripherals](https://www.raspberrypi.org/app/uploads/2012/02/BCM2835-ARM-Peripherals.pdf)
* [VideoCore IV Programmers Manual](https://docs.broadcom.com/doc/12358545)
* [Raspberry Pi Mailbox Property Interface](https://github.com/raspberrypi/firmware/wiki/Mailbox-property-interface)

## 📜 License

MIT License — See repository root for details.
