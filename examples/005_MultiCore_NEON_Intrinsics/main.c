/**
 * 005_MultiCore_NEON_Intrinsics - main.c
 * 
 * Benchmark driver for high-performance matrix multiplication on Raspberry Pi 3B.
 * 
 * This program compares three implementations:
 *   1. Naive triple-loop (baseline)
 *   2. NEON intrinsics (single-threaded)
 *   3. NEON intrinsics + OpenMP (multi-threaded)
 * 
 * Usage: ./matmul_neon_omp [matrix_size]
 *        Default: 1024
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "matmul_neon_omp.h"

/* ============================================================================
 * Configuration
 * ============================================================================ */

#define DEFAULT_SIZE    1024
#define EPSILON         1e-4f
#define NUM_WARMUP      1
#define NUM_ITERATIONS  3

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ============================================================================
 * Matrix Utilities
 * ============================================================================ */

static float *alloc_matrix(int n) {
    /*
     * Allocate 16-byte aligned memory for optimal NEON performance.
     * posix_memalign ensures the pointer is aligned to a 16-byte boundary.
     */
    float *mat;
    if (posix_memalign((void **)&mat, 16, n * n * sizeof(float)) != 0) {
        fprintf(stderr, "Memory allocation failed for %dx%d matrix\n", n, n);
        return NULL;
    }
    return mat;
}

static void init_matrix_random(float *mat, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < n * n; i++) {
        /* Random values in [-1, 1] */
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

static void init_matrix_zero(float *mat, int n) {
    memset(mat, 0, n * n * sizeof(float));
}

static int verify_result(const float *C_test, const float *C_ref, int n, 
                         float epsilon, float *max_err) {
    *max_err = 0.0f;
    int pass = 1;
    
    for (int i = 0; i < n * n; i++) {
        float err = fabsf(C_test[i] - C_ref[i]);
        if (err > *max_err) {
            *max_err = err;
        }
        if (err > epsilon) {
            pass = 0;
        }
    }
    
    return pass;
}

/* ============================================================================
 * Benchmark Function
 * ============================================================================ */

typedef void (*matmul_func)(const float *, const float *, float *, int);

static double benchmark(matmul_func func, const float *A, const float *B, 
                        float *C, int n, int warmup, int iterations) {
    /* Warmup runs (not timed) */
    for (int i = 0; i < warmup; i++) {
        func(A, B, C, n);
    }
    
    /* Timed runs */
    double total_time = 0.0;
    for (int i = 0; i < iterations; i++) {
        init_matrix_zero(C, n);
        
        double t0 = get_time_sec();
        func(A, B, C, n);
        double t1 = get_time_sec();
        
        total_time += (t1 - t0);
    }
    
    return total_time / iterations;
}

/* ============================================================================
 * Print Utilities
 * ============================================================================ */

static void print_header(void) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║        005_MultiCore_NEON_Intrinsics - Matrix Multiplication         ║\n");
    printf("║                     Raspberry Pi 3B (Cortex-A53)                     ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

static void print_system_info(void) {
    int num_threads = get_num_threads();
    
    printf("System Information:\n");
    printf("  CPU:             Cortex-A53 @ 1.4 GHz (estimated)\n");
    printf("  OpenMP Threads:  %d\n", num_threads);
    printf("  SIMD:            ARM NEON (128-bit, 4×float)\n");
    printf("\n");
}

static void print_theoretical_peak(int n) {
    /*
     * Theoretical peak calculation for Cortex-A53:
     * - NEON can issue 1 FMA per cycle (2 FLOPs) per 4 floats = 8 FLOP/cycle
     * - At 1.4 GHz: 11.2 GFLOPS per core
     * - 4 cores: 44.8 GFLOPS theoretical maximum
     * 
     * Matrix multiply FLOPs: 2 * n^3 (n^3 multiplies + n^3 adds)
     */
    double flops = 2.0 * (double)n * (double)n * (double)n;
    
    printf("Workload:\n");
    printf("  Matrix Size:     %d × %d\n", n, n);
    printf("  FLOPs:           %.2f GFLOP\n", flops / 1e9);
    printf("  Memory:          %.2f MB (3 matrices)\n", 3.0 * n * n * sizeof(float) / (1024 * 1024));
    printf("\n");
    
    printf("Theoretical Peak Performance (Cortex-A53 @ 1.4 GHz):\n");
    printf("  Single Core:     11.2 GFLOPS\n");
    printf("  Quad Core:       44.8 GFLOPS\n");
    printf("  Note: Memory bandwidth typically limits to 30-50%% of peak.\n");
    printf("\n");
}

static void print_result(const char *name, double time_sec, int n, int threads,
                         int pass, float max_err) {
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double gflops = flops / time_sec / 1e9;
    
    printf("  %-30s  %8.3f sec  %7.2f GFLOPS  [%s] (err=%.2e)\n",
           name, time_sec, gflops, pass ? "PASS" : "FAIL", max_err);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(int argc, char *argv[]) {
    /* Parse command line */
    int n = DEFAULT_SIZE;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
            return 1;
        }
    }
    
    /* Ensure n is a multiple of 4 for NEON alignment */
    if (n % 4 != 0) {
        fprintf(stderr, "Warning: Matrix size %d is not a multiple of 4. "
                       "Rounding up to %d.\n", n, ((n + 3) / 4) * 4);
        n = ((n + 3) / 4) * 4;
    }
    
    print_header();
    print_system_info();
    print_theoretical_peak(n);
    
    /* Allocate matrices */
    printf("Allocating matrices...\n");
    float *A = alloc_matrix(n);
    float *B = alloc_matrix(n);
    float *C_naive = alloc_matrix(n);
    float *C_neon = alloc_matrix(n);
    float *C_neon_omp = alloc_matrix(n);
    
    if (!A || !B || !C_naive || !C_neon || !C_neon_omp) {
        fprintf(stderr, "Memory allocation failed!\n");
        return 1;
    }
    
    /* Initialize matrices with reproducible random values */
    printf("Initializing matrices with random values...\n\n");
    init_matrix_random(A, n, 42);
    init_matrix_random(B, n, 123);
    
    /* Run benchmarks */
    printf("Running benchmarks (%d warmup, %d iterations each):\n\n",
           NUM_WARMUP, NUM_ITERATIONS);
    
    double time_naive, time_neon, time_neon_omp;
    float max_err;
    int pass;
    
    /* 1. Naive implementation */
    printf("  [1/3] Naive triple-loop (single thread)...\n");
    time_naive = benchmark(matmul_naive, A, B, C_naive, n, 0, 1);  /* Just 1 iteration */
    printf("        Done.\n");
    
    /* 2. NEON single-threaded */
    printf("  [2/3] NEON intrinsics (single thread)...\n");
    time_neon = benchmark(matmul_neon_single, A, B, C_neon, n, NUM_WARMUP, NUM_ITERATIONS);
    pass = verify_result(C_neon, C_naive, n, EPSILON, &max_err);
    printf("        Done.\n");
    
    /* 3. NEON + OpenMP */
    printf("  [3/3] NEON intrinsics + OpenMP (%d threads)...\n", get_num_threads());
    time_neon_omp = benchmark(matmul_neon_omp, A, B, C_neon_omp, n, NUM_WARMUP, NUM_ITERATIONS);
    int pass_omp = verify_result(C_neon_omp, C_naive, n, EPSILON, &max_err);
    printf("        Done.\n\n");
    
    /* Print results */
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("Results (Matrix: %d×%d, epsilon=%.0e):\n\n", n, n, EPSILON);
    
    /* Re-verify for reporting */
    verify_result(C_neon, C_naive, n, EPSILON, &max_err);
    print_result("Naive (1 thread)", time_naive, n, 1, 1, 0.0f);
    print_result("NEON (1 thread)", time_neon, n, 1, pass, max_err);
    
    verify_result(C_neon_omp, C_naive, n, EPSILON, &max_err);
    print_result("NEON+OpenMP (4 threads)", time_neon_omp, n, get_num_threads(), pass_omp, max_err);
    
    printf("\n");
    
    /* Speedup analysis */
    printf("Speedup Analysis:\n");
    printf("  NEON vs Naive:           %.2fx\n", time_naive / time_neon);
    printf("  NEON+OMP vs Naive:       %.2fx\n", time_naive / time_neon_omp);
    printf("  NEON+OMP vs NEON:        %.2fx (parallel efficiency: %.0f%%)\n", 
           time_neon / time_neon_omp,
           100.0 * (time_neon / time_neon_omp) / get_num_threads());
    printf("\n");
    
    /* Performance analysis */
    double flops = 2.0 * (double)n * (double)n * (double)n;
    double peak_single = 11.2;  /* GFLOPS */
    double peak_quad = 44.8;
    
    printf("Efficiency vs Theoretical Peak:\n");
    printf("  NEON (1 thread):         %.1f%% of single-core peak (%.1f GFLOPS)\n",
           100.0 * (flops / time_neon / 1e9) / peak_single,
           flops / time_neon / 1e9);
    printf("  NEON+OMP (4 threads):    %.1f%% of quad-core peak (%.1f GFLOPS)\n",
           100.0 * (flops / time_neon_omp / 1e9) / peak_quad,
           flops / time_neon_omp / 1e9);
    printf("\n");
    
    /* Final verdict */
    printf("══════════════════════════════════════════════════════════════════════════\n");
    if (pass && pass_omp) {
        printf("All tests PASSED.\n");
    } else {
        printf("Some tests FAILED!\n");
    }
    printf("══════════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    
    /* Cleanup */
    free(A);
    free(B);
    free(C_naive);
    free(C_neon);
    free(C_neon_omp);
    
    return (pass && pass_omp) ? 0 : 1;
}
