/**
 * 006_Zero_Copy_Shared_Memory - main.c
 * 
 * Micro-benchmark demonstrating the bandwidth benefits of Zero-Copy architecture
 * on the Raspberry Pi 3B with VideoCore IV GPU.
 * 
 * This benchmark compares two approaches for getting data to GPU-accessible memory:
 * 
 * 1. STANDARD (Copy) Approach:
 *    - malloc() a buffer in CPU RAM (cached)
 *    - mem_alloc() a buffer in GPU RAM
 *    - Fill CPU buffer (fast, cached writes)
 *    - memcpy() CPU buffer → GPU buffer (copy overhead)
 * 
 * 2. ZERO-COPY (Direct) Approach:
 *    - mem_alloc() a buffer in GPU RAM with DIRECT flag
 *    - mmap() it to user space (uncached)
 *    - Write directly to the mapped buffer (slower writes, but no copy)
 * 
 * Key Insight:
 * ============
 * Writing to uncached memory is slower per-byte than writing to cached memory.
 * However, the Standard approach pays a DOUBLE tax:
 *   1. Write to cached CPU memory
 *   2. Copy from CPU memory to GPU memory
 * 
 * The Zero-Copy approach pays only ONCE but at a slower rate.
 * For large transfers, eliminating the copy often wins.
 * 
 * Target: Raspberry Pi 3B (BCM2837, VideoCore IV)
 * Author: HPC_GPGPU Course
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <errno.h>

#include "mailbox.h"

/* ============================================================================
 * Configuration
 * ============================================================================ */

/** Default data size: 64 MB (large enough to see real bandwidth effects) */
#define DEFAULT_DATA_SIZE   (64 * 1024 * 1024)

/** Alignment for GPU memory (page size) */
#define GPU_ALIGNMENT       4096

/** Number of warmup iterations */
#define NUM_WARMUP          1

/** Number of timed iterations */
#define NUM_ITERATIONS      5

/** Pattern for verification */
#define FILL_PATTERN        0xDEADBEEF


/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

/**
 * @brief Get current time in microseconds.
 */
static inline uint64_t get_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

/**
 * @brief Get current time in nanoseconds (higher resolution).
 */
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}


/* ============================================================================
 * Data Generation Functions
 * ============================================================================
 * 
 * We use NEON-optimized fills where available for realistic workloads.
 * The fill pattern simulates real data generation (not just memset).
 */

/**
 * @brief Fill buffer with incrementing 32-bit pattern.
 * 
 * This simulates a realistic data generation workload where each word
 * has a unique value (useful for verification).
 */
static void fill_buffer_pattern(uint32_t *buf, size_t size_bytes, uint32_t seed) {
    size_t num_words = size_bytes / sizeof(uint32_t);
    uint32_t value = seed;
    
    for (size_t i = 0; i < num_words; i++) {
        buf[i] = value++;
    }
}

/**
 * @brief Fill buffer using NEON-optimized stores (if available).
 * 
 * Uses 128-bit stores for maximum memory bandwidth.
 */
#ifdef __ARM_NEON
#include <arm_neon.h>

static void fill_buffer_neon(uint32_t *buf, size_t size_bytes, uint32_t pattern) {
    size_t num_vectors = size_bytes / (4 * sizeof(uint32_t));
    uint32x4_t vec_pattern = vdupq_n_u32(pattern);
    
    for (size_t i = 0; i < num_vectors; i++) {
        vst1q_u32(buf + i * 4, vec_pattern);
    }
    
    /* Handle remainder */
    size_t remainder_start = num_vectors * 4;
    size_t num_words = size_bytes / sizeof(uint32_t);
    for (size_t i = remainder_start; i < num_words; i++) {
        buf[i] = pattern;
    }
}
#else
static void fill_buffer_neon(uint32_t *buf, size_t size_bytes, uint32_t pattern) {
    /* Fallback for non-NEON builds */
    size_t num_words = size_bytes / sizeof(uint32_t);
    for (size_t i = 0; i < num_words; i++) {
        buf[i] = pattern;
    }
}
#endif

/* Attribute to silence unused warning - function available for alternative benchmarks */
static void fill_buffer_neon_warmup(uint32_t *buf, size_t size_bytes, uint32_t pattern) 
    __attribute__((unused));
static void fill_buffer_neon_warmup(uint32_t *buf, size_t size_bytes, uint32_t pattern) {
    fill_buffer_neon(buf, size_bytes, pattern);
}


/* ============================================================================
 * Verification Functions
 * ============================================================================ */

/**
 * @brief Verify buffer contains expected incrementing pattern.
 */
static int verify_buffer_pattern(const uint32_t *buf, size_t size_bytes, 
                                  uint32_t seed, const char *name) {
    size_t num_words = size_bytes / sizeof(uint32_t);
    uint32_t expected = seed;
    int errors = 0;
    
    for (size_t i = 0; i < num_words; i++) {
        if (buf[i] != expected) {
            if (errors < 10) {
                fprintf(stderr, "[%s] Mismatch at word %zu: expected 0x%08X, got 0x%08X\n",
                        name, i, expected, buf[i]);
            }
            errors++;
        }
        expected++;
    }
    
    if (errors > 0) {
        fprintf(stderr, "[%s] Total errors: %d / %zu words\n", name, errors, num_words);
    }
    
    return errors == 0;
}


/* ============================================================================
 * Benchmark Functions
 * ============================================================================ */

/**
 * @brief Benchmark 1: Standard Copy Approach
 * 
 * 1. malloc() in CPU RAM (cached)
 * 2. mem_alloc() in GPU RAM
 * 3. Fill CPU buffer
 * 4. memcpy() to GPU buffer
 */
typedef struct {
    double total_time_ms;       /* Total time (fill + copy) */
    double fill_time_ms;        /* Time to fill CPU buffer */
    double copy_time_ms;        /* Time to memcpy to GPU buffer */
    double fill_bandwidth_gbps; /* Fill bandwidth */
    double copy_bandwidth_gbps; /* Copy bandwidth */
    double total_bandwidth_gbps;/* Effective bandwidth */
    int verified;               /* Verification passed */
} bench_result_standard_t;

static int benchmark_standard_copy(mbox_handle_t mbox, size_t data_size,
                                    int warmup, int iterations,
                                    bench_result_standard_t *result) {
    printf("\n  Running Standard Copy benchmark...\n");
    
    /* Allocate CPU buffer (cached) */
    uint32_t *cpu_buf = (uint32_t *)aligned_alloc(GPU_ALIGNMENT, data_size);
    if (!cpu_buf) {
        fprintf(stderr, "  Error: Failed to allocate CPU buffer\n");
        return -1;
    }
    
    /* Allocate GPU buffer (coherent for fair comparison) */
    gpu_mem_t gpu_mem;
    if (gpu_mem_alloc(mbox, data_size, GPU_ALIGNMENT, 
                      MEM_FLAG_COHERENT | MEM_FLAG_ZERO, &gpu_mem) < 0) {
        fprintf(stderr, "  Error: Failed to allocate GPU buffer\n");
        free(cpu_buf);
        return -1;
    }
    
    printf("    CPU buffer: %p (cached)\n", cpu_buf);
    printf("    GPU buffer: %p (bus: 0x%08X, coherent)\n", 
           gpu_mem.virt_addr, gpu_mem.bus_addr);
    
    /* Warmup runs */
    for (int i = 0; i < warmup; i++) {
        fill_buffer_pattern(cpu_buf, data_size, 0);
        memcpy(gpu_mem.virt_addr, cpu_buf, data_size);
    }
    
    /* Timed runs */
    double total_fill_time = 0.0;
    double total_copy_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        /* Clear GPU buffer */
        memset(gpu_mem.virt_addr, 0, data_size);
        
        /* Memory barrier to ensure previous writes complete */
        __sync_synchronize();
        
        /* Time: Fill CPU buffer */
        uint64_t t0 = get_time_ns();
        fill_buffer_pattern(cpu_buf, data_size, 0x12345678);
        __sync_synchronize();
        uint64_t t1 = get_time_ns();
        
        /* Time: Copy to GPU buffer */
        memcpy(gpu_mem.virt_addr, cpu_buf, data_size);
        __sync_synchronize();
        uint64_t t2 = get_time_ns();
        
        total_fill_time += (t1 - t0) / 1e6;  /* Convert to ms */
        total_copy_time += (t2 - t1) / 1e6;
    }
    
    /* Calculate averages */
    double avg_fill_ms = total_fill_time / iterations;
    double avg_copy_ms = total_copy_time / iterations;
    double avg_total_ms = avg_fill_ms + avg_copy_ms;
    
    /* Calculate bandwidths in GB/s */
    double data_gb = (double)data_size / (1024.0 * 1024.0 * 1024.0);
    
    result->fill_time_ms = avg_fill_ms;
    result->copy_time_ms = avg_copy_ms;
    result->total_time_ms = avg_total_ms;
    result->fill_bandwidth_gbps = data_gb / (avg_fill_ms / 1000.0);
    result->copy_bandwidth_gbps = data_gb / (avg_copy_ms / 1000.0);
    result->total_bandwidth_gbps = data_gb / (avg_total_ms / 1000.0);
    
    /* Verify */
    result->verified = verify_buffer_pattern((const uint32_t *)gpu_mem.virt_addr,
                                              data_size, 0x12345678, "Standard");
    
    /* Cleanup */
    gpu_mem_free(&gpu_mem);
    free(cpu_buf);
    
    return 0;
}


/**
 * @brief Benchmark 2: Zero-Copy Direct Approach
 * 
 * 1. mem_alloc() with MEM_FLAG_DIRECT (uncached)
 * 2. mmap() to user space
 * 3. Write directly to mapped buffer (no copy needed)
 */
typedef struct {
    double total_time_ms;       /* Total time (direct write only) */
    double write_bandwidth_gbps;/* Write bandwidth */
    int verified;               /* Verification passed */
} bench_result_zerocopy_t;

static int benchmark_zero_copy(mbox_handle_t mbox, size_t data_size,
                                int warmup, int iterations,
                                bench_result_zerocopy_t *result) {
    printf("\n  Running Zero-Copy Direct benchmark...\n");
    
    /* Allocate GPU buffer with DIRECT flag (uncached) */
    gpu_mem_t gpu_mem;
    if (gpu_mem_alloc(mbox, data_size, GPU_ALIGNMENT, 
                      MEM_FLAG_DIRECT | MEM_FLAG_ZERO, &gpu_mem) < 0) {
        fprintf(stderr, "  Error: Failed to allocate GPU buffer (direct)\n");
        return -1;
    }
    
    printf("    GPU buffer: %p (bus: 0x%08X, direct/uncached)\n", 
           gpu_mem.virt_addr, gpu_mem.bus_addr);
    printf("    Alias: 0x%X (expected: 0xC for direct)\n", 
           bus_get_alias(gpu_mem.bus_addr));
    
    /* Warmup runs */
    for (int i = 0; i < warmup; i++) {
        fill_buffer_pattern((uint32_t *)gpu_mem.virt_addr, data_size, 0);
    }
    
    /* Timed runs */
    double total_write_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        /* Clear buffer */
        memset(gpu_mem.virt_addr, 0, data_size);
        
        /* Memory barrier */
        __sync_synchronize();
        
        /* Time: Direct write to GPU buffer */
        uint64_t t0 = get_time_ns();
        fill_buffer_pattern((uint32_t *)gpu_mem.virt_addr, data_size, 0x12345678);
        __sync_synchronize();
        uint64_t t1 = get_time_ns();
        
        total_write_time += (t1 - t0) / 1e6;
    }
    
    /* Calculate averages */
    double avg_write_ms = total_write_time / iterations;
    double data_gb = (double)data_size / (1024.0 * 1024.0 * 1024.0);
    
    result->total_time_ms = avg_write_ms;
    result->write_bandwidth_gbps = data_gb / (avg_write_ms / 1000.0);
    
    /* Verify */
    result->verified = verify_buffer_pattern((const uint32_t *)gpu_mem.virt_addr,
                                              data_size, 0x12345678, "ZeroCopy");
    
    /* Cleanup */
    gpu_mem_free(&gpu_mem);
    
    return 0;
}


/**
 * @brief Benchmark 3: Baseline - cached malloc write speed
 * 
 * This measures the CPU's raw write speed to cached memory for reference.
 */
typedef struct {
    double time_ms;
    double bandwidth_gbps;
} bench_result_baseline_t;

static int benchmark_baseline_cached(size_t data_size, int iterations,
                                      bench_result_baseline_t *result) {
    printf("\n  Running Baseline (cached malloc) benchmark...\n");
    
    uint32_t *buf = (uint32_t *)aligned_alloc(GPU_ALIGNMENT, data_size);
    if (!buf) {
        fprintf(stderr, "  Error: Failed to allocate buffer\n");
        return -1;
    }
    
    /* Warmup */
    fill_buffer_pattern(buf, data_size, 0);
    
    /* Timed runs */
    double total_time = 0.0;
    
    for (int i = 0; i < iterations; i++) {
        memset(buf, 0, data_size);
        __sync_synchronize();
        
        uint64_t t0 = get_time_ns();
        fill_buffer_pattern(buf, data_size, 0x12345678);
        __sync_synchronize();
        uint64_t t1 = get_time_ns();
        
        total_time += (t1 - t0) / 1e6;
    }
    
    double avg_ms = total_time / iterations;
    double data_gb = (double)data_size / (1024.0 * 1024.0 * 1024.0);
    
    result->time_ms = avg_ms;
    result->bandwidth_gbps = data_gb / (avg_ms / 1000.0);
    
    free(buf);
    return 0;
}


/* ============================================================================
 * Print Utilities
 * ============================================================================ */

static void print_header(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║          006_Zero_Copy_Shared_Memory - Bandwidth Benchmark           ║\n");
    printf("║                     Raspberry Pi 3B (BCM2837)                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_system_info(mbox_handle_t mbox) {
    printf("System Information:\n");
    
    uint32_t fw_version = get_firmware_version(mbox);
    printf("  Firmware:    0x%08X\n", fw_version);
    
    uint32_t arm_base, arm_size;
    if (get_arm_memory(mbox, &arm_base, &arm_size) == 0) {
        printf("  ARM Memory:  0x%08X - 0x%08X (%u MB)\n", 
               arm_base, arm_base + arm_size - 1, arm_size / (1024 * 1024));
    }
    
    uint32_t vc_base, vc_size;
    if (get_vc_memory(mbox, &vc_base, &vc_size) == 0) {
        printf("  GPU Memory:  0x%08X - 0x%08X (%u MB)\n",
               vc_base, vc_base + vc_size - 1, vc_size / (1024 * 1024));
    }
    
    printf("\n");
}

static void print_memory_aliases(void) {
    printf("Memory Address Aliases (BCM2837):\n");
    printf("  ┌─────────┬──────────────┬─────────────────────────────┐\n");
    printf("  │ Alias   │ Base Address │ Caching                     │\n");
    printf("  ├─────────┼──────────────┼─────────────────────────────┤\n");
    printf("  │ 0x0     │ 0x00000000   │ L1 & L2 cached              │\n");
    printf("  │ 0x4     │ 0x40000000   │ L2 coherent (ARM visible)   │\n");
    printf("  │ 0x8     │ 0x80000000   │ L2 cached (allocating)      │\n");
    printf("  │ 0xC     │ 0xC0000000   │ Direct/Uncached (bypass)    │\n");
    printf("  └─────────┴──────────────┴─────────────────────────────┘\n");
    printf("\n");
}


/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char *argv[]) {
    /* Parse command line */
    size_t data_size = DEFAULT_DATA_SIZE;
    
    if (argc > 1) {
        int mb = atoi(argv[1]);
        if (mb > 0 && mb <= 256) {
            data_size = (size_t)mb * 1024 * 1024;
        } else {
            fprintf(stderr, "Invalid size. Using default: %zu MB\n", 
                    DEFAULT_DATA_SIZE / (1024 * 1024));
        }
    }
    
    print_header();
    
    printf("Configuration:\n");
    printf("  Data Size:   %zu MB (%zu bytes)\n", 
           data_size / (1024 * 1024), data_size);
    printf("  Warmup:      %d iterations\n", NUM_WARMUP);
    printf("  Iterations:  %d (averaged)\n", NUM_ITERATIONS);
    printf("\n");
    
    /* Open mailbox */
    mbox_handle_t mbox = mbox_open();
    if (mbox == MBOX_INVALID_HANDLE) {
        fprintf(stderr, "Error: Cannot open mailbox. Run as root.\n");
        return 1;
    }
    
    print_system_info(mbox);
    print_memory_aliases();
    
    /* Run benchmarks */
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("Running Benchmarks...\n");
    
    bench_result_baseline_t baseline_result = {0};
    bench_result_standard_t standard_result = {0};
    bench_result_zerocopy_t zerocopy_result = {0};
    
    int baseline_ok = benchmark_baseline_cached(data_size, NUM_ITERATIONS, 
                                                 &baseline_result);
    int standard_ok = benchmark_standard_copy(mbox, data_size, 
                                               NUM_WARMUP, NUM_ITERATIONS, 
                                               &standard_result);
    int zerocopy_ok = benchmark_zero_copy(mbox, data_size, 
                                           NUM_WARMUP, NUM_ITERATIONS, 
                                           &zerocopy_result);
    
    /* Print Results */
    printf("\n══════════════════════════════════════════════════════════════════════════\n");
    printf("Results Summary (%zu MB data):\n\n", data_size / (1024 * 1024));
    
    printf("  ┌────────────────────────────┬─────────────┬─────────────┬──────────┐\n");
    printf("  │ Benchmark                  │ Time (ms)   │ BW (GB/s)   │ Status   │\n");
    printf("  ├────────────────────────────┼─────────────┼─────────────┼──────────┤\n");
    
    if (baseline_ok == 0) {
        printf("  │ Baseline (cached malloc)   │ %9.2f   │ %9.3f   │  REF     │\n",
               baseline_result.time_ms, baseline_result.bandwidth_gbps);
    }
    
    if (standard_ok == 0) {
        printf("  ├────────────────────────────┼─────────────┼─────────────┼──────────┤\n");
        printf("  │ Standard: Fill (cached)    │ %9.2f   │ %9.3f   │          │\n",
               standard_result.fill_time_ms, standard_result.fill_bandwidth_gbps);
        printf("  │ Standard: Copy (memcpy)    │ %9.2f   │ %9.3f   │          │\n",
               standard_result.copy_time_ms, standard_result.copy_bandwidth_gbps);
        printf("  │ Standard: TOTAL            │ %9.2f   │ %9.3f   │ [%s]  │\n",
               standard_result.total_time_ms, standard_result.total_bandwidth_gbps,
               standard_result.verified ? "PASS" : "FAIL");
    }
    
    if (zerocopy_ok == 0) {
        printf("  ├────────────────────────────┼─────────────┼─────────────┼──────────┤\n");
        printf("  │ Zero-Copy: Direct write    │ %9.2f   │ %9.3f   │ [%s]  │\n",
               zerocopy_result.total_time_ms, zerocopy_result.write_bandwidth_gbps,
               zerocopy_result.verified ? "PASS" : "FAIL");
    }
    
    printf("  └────────────────────────────┴─────────────┴─────────────┴──────────┘\n");
    
    /* Analysis */
    if (standard_ok == 0 && zerocopy_ok == 0) {
        printf("\n");
        printf("Analysis:\n");
        
        double speedup = standard_result.total_time_ms / zerocopy_result.total_time_ms;
        double copy_overhead_pct = 100.0 * standard_result.copy_time_ms / 
                                   standard_result.total_time_ms;
        
        if (speedup > 1.0) {
            printf("  ✓ Zero-Copy is %.2fx FASTER than Standard approach\n", speedup);
        } else {
            printf("  ✗ Zero-Copy is %.2fx SLOWER than Standard approach\n", 1.0/speedup);
        }
        
        printf("  • Copy overhead in Standard: %.1f%% of total time\n", copy_overhead_pct);
        double write_ratio = zerocopy_result.write_bandwidth_gbps / baseline_result.bandwidth_gbps;
        if (write_ratio < 1.0) {
            printf("  • Uncached write penalty: %.2fx slower than cached\n", 1.0/write_ratio);
        } else {
            printf("  • Uncached write bonus: %.2fx faster than cached baseline!\n", write_ratio);
        }
        
        printf("\n");
        printf("Key Insights:\n");
        
        if (baseline_result.bandwidth_gbps > zerocopy_result.write_bandwidth_gbps) {
            printf("  1. Cached writes (%.2f GB/s) are faster than uncached (%.2f GB/s)\n",
                   baseline_result.bandwidth_gbps, zerocopy_result.write_bandwidth_gbps);
            printf("  2. But Standard pays TWICE: fill + copy\n");
            printf("  3. Zero-Copy pays ONCE: direct write (no copy overhead)\n");
        } else {
            printf("  1. Surprisingly, uncached writes (%.2f GB/s) >= cached (%.2f GB/s)!\n",
                   zerocopy_result.write_bandwidth_gbps, baseline_result.bandwidth_gbps);
            printf("  2. This suggests write-combining or store buffers are effective\n");
            printf("  3. Standard STILL pays copy overhead (%.1f%% of time)\n", copy_overhead_pct);
        }
        
        if (speedup > 1.0) {
            printf("  4. For %zu MB transfers, eliminating the copy wins!\n",
                   data_size / (1024 * 1024));
        } else {
            printf("  4. For smaller transfers, cache benefits might outweigh copy cost\n");
        }
    }
    
    printf("\n══════════════════════════════════════════════════════════════════════════\n");
    
    /* Cleanup */
    mbox_close(mbox);
    
    return 0;
}
