/**
 * matmul_neon_omp.h
 * 
 * Header file for high-performance matrix multiplication using
 * ARM NEON intrinsics and OpenMP parallelization.
 * 
 * Target: Raspberry Pi 3B (Cortex-A53, 4 cores)
 */

#ifndef MATMUL_NEON_OMP_H
#define MATMUL_NEON_OMP_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief High-performance matrix multiplication using NEON + OpenMP.
 * 
 * Computes C = A × B where all matrices are n×n single-precision floats.
 * Uses 4×4 register blocking to maximize NEON utilization and OpenMP
 * to distribute work across all available cores.
 * 
 * @param A     Input matrix A (n×n, row-major)
 * @param B     Input matrix B (n×n, row-major) - will be transposed internally
 * @param C     Output matrix C (n×n, row-major)
 * @param n     Matrix dimension (must be multiple of 4)
 * 
 * @note Matrix dimension n must be a multiple of 4 for NEON alignment.
 * @note Matrices should be 16-byte aligned for optimal performance.
 * @note This function will transpose B internally for cache-friendly access.
 */
void matmul_neon_omp(const float *A, const float *B, float *C, int n);

/**
 * @brief NEON-optimized matrix multiplication (single-threaded).
 * 
 * Same as matmul_neon_omp but without OpenMP parallelization.
 * Useful for benchmarking the NEON speedup independent of threading.
 * 
 * @param A     Input matrix A (n×n, row-major)
 * @param B     Input matrix B (n×n, row-major) - will be transposed internally
 * @param C     Output matrix C (n×n, row-major)
 * @param n     Matrix dimension (must be multiple of 4)
 */
void matmul_neon_single(const float *A, const float *B, float *C, int n);

/**
 * @brief Naive triple-loop matrix multiplication (reference implementation).
 * 
 * Simple O(n³) algorithm for correctness verification.
 * No SIMD, no parallelization.
 * 
 * @param A     Input matrix A (n×n, row-major)
 * @param B     Input matrix B (n×n, row-major)
 * @param C     Output matrix C (n×n, row-major)
 * @param n     Matrix dimension
 */
void matmul_naive(const float *A, const float *B, float *C, int n);

/**
 * @brief Get the number of OpenMP threads that will be used.
 * 
 * @return Number of threads (typically 4 on Raspberry Pi 3B)
 */
int get_num_threads(void);

/**
 * @brief Transpose a matrix in-place (for square matrices) or out-of-place.
 * 
 * Converts B[i][j] to B_T[j][i] for cache-friendly column access.
 * 
 * @param src   Source matrix (n×n)
 * @param dst   Destination matrix (n×n), can be same as src for in-place
 * @param n     Matrix dimension
 */
void transpose_matrix(const float *src, float *dst, int n);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_NEON_OMP_H */
