/**
 * matmul_neon_omp_v5.c
 * 
 * CACHE-TILED version with L2-friendly blocking.
 * 
 * Key insight: v1 achieved 0.81 GFLOPS but we're at only 7% of peak.
 * The bottleneck is memory bandwidth - matrices don't fit in cache.
 * 
 * Solution: TILING (blocking)
 * - Divide matrices into tiles that fit in L2 cache
 * - Process tiles to maximize data reuse before eviction
 * 
 * Raspberry Pi 3B cache hierarchy:
 * - L1 Data: 32 KB per core (8-way, 64-byte lines)
 * - L2: 512 KB shared (16-way)
 * 
 * Tile size calculation:
 * - For C[i:i+T][j:j+T] += A[i:i+T][k:k+T] × B[k:k+T][j:j+T]
 * - Need: T² floats from A + T² floats from B + T² floats for C
 * - Total: 3 × T² × 4 bytes = 12T² bytes
 * - For L2 (512KB): T² ≤ 512KB/12 → T ≤ 207
 * - Use T = 64 or 128 for good alignment and to leave room for other data
 * 
 * With T=64: 3 × 64² × 4 = 48KB (fits easily in L2, leaves room for BT)
 */

#include "matmul_neon_omp.h"
#include <arm_neon.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* Tile size - must be multiple of 4 for NEON alignment */
#define TILE_SIZE 64

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

int get_num_threads(void) {
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    return num_threads;
}

void transpose_matrix(const float *src, float *dst, int n) {
    int i, j;
    for (i = 0; i <= n - 4; i += 4) {
        for (j = 0; j <= n - 4; j += 4) {
            float32x4_t r0 = vld1q_f32(&src[(i + 0) * n + j]);
            float32x4_t r1 = vld1q_f32(&src[(i + 1) * n + j]);
            float32x4_t r2 = vld1q_f32(&src[(i + 2) * n + j]);
            float32x4_t r3 = vld1q_f32(&src[(i + 3) * n + j]);
            
            float32x4x2_t t01 = vtrnq_f32(r0, r1);
            float32x4x2_t t23 = vtrnq_f32(r2, r3);
            
            float32x4_t c0 = vcombine_f32(vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
            float32x4_t c1 = vcombine_f32(vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
            float32x4_t c2 = vcombine_f32(vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
            float32x4_t c3 = vcombine_f32(vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
            
            vst1q_f32(&dst[(j + 0) * n + i], c0);
            vst1q_f32(&dst[(j + 1) * n + i], c1);
            vst1q_f32(&dst[(j + 2) * n + i], c2);
            vst1q_f32(&dst[(j + 3) * n + i], c3);
        }
        for (; j < n; j++) {
            for (int ii = i; ii < i + 4; ii++) {
                dst[j * n + ii] = src[ii * n + j];
            }
        }
    }
    for (; i < n; i++) {
        for (j = 0; j < n; j++) {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

/* ============================================================================
 * Naive Reference Implementation
 * ============================================================================ */

void matmul_naive(const float *A, const float *B, float *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

/* ============================================================================
 * 4×4 Micro-kernel (same as v1, proven fastest)
 * ============================================================================
 * 
 * Uses vmlaq_lane_f32 with in-register transpose of B tile.
 * This was the fastest approach in v1 (0.81 GFLOPS).
 */

static inline void kernel_4x4_neon(
    const float * restrict A,
    const float * restrict BT,
    float * restrict C,
    int lda,    /* leading dimension of A */
    int ldbt,   /* leading dimension of BT */
    int ldc,    /* leading dimension of C */
    int K       /* inner dimension */
) {
    float32x4_t c_row0 = vld1q_f32(C);
    float32x4_t c_row1 = vld1q_f32(C + ldc);
    float32x4_t c_row2 = vld1q_f32(C + 2 * ldc);
    float32x4_t c_row3 = vld1q_f32(C + 3 * ldc);
    
    const float *a_row0 = A;
    const float *a_row1 = A + lda;
    const float *a_row2 = A + 2 * lda;
    const float *a_row3 = A + 3 * lda;
    
    const float *bt_row0 = BT;
    const float *bt_row1 = BT + ldbt;
    const float *bt_row2 = BT + 2 * ldbt;
    const float *bt_row3 = BT + 3 * ldbt;
    
    int k = 0;
    for (; k <= K - 4; k += 4) {
        float32x4_t a0 = vld1q_f32(a_row0 + k);
        float32x4_t a1 = vld1q_f32(a_row1 + k);
        float32x4_t a2 = vld1q_f32(a_row2 + k);
        float32x4_t a3 = vld1q_f32(a_row3 + k);
        
        float32x4_t b0 = vld1q_f32(bt_row0 + k);
        float32x4_t b1 = vld1q_f32(bt_row1 + k);
        float32x4_t b2 = vld1q_f32(bt_row2 + k);
        float32x4_t b3 = vld1q_f32(bt_row3 + k);
        
        /* 4×4 transpose of b vectors */
        float32x4x2_t b01 = vtrnq_f32(b0, b1);
        float32x4x2_t b23 = vtrnq_f32(b2, b3);
        
        float32x4_t bt0 = vcombine_f32(vget_low_f32(b01.val[0]), vget_low_f32(b23.val[0]));
        float32x4_t bt1 = vcombine_f32(vget_low_f32(b01.val[1]), vget_low_f32(b23.val[1]));
        float32x4_t bt2 = vcombine_f32(vget_high_f32(b01.val[0]), vget_high_f32(b23.val[0]));
        float32x4_t bt3 = vcombine_f32(vget_high_f32(b01.val[1]), vget_high_f32(b23.val[1]));
        
        float32x2_t a0_lo = vget_low_f32(a0), a0_hi = vget_high_f32(a0);
        float32x2_t a1_lo = vget_low_f32(a1), a1_hi = vget_high_f32(a1);
        float32x2_t a2_lo = vget_low_f32(a2), a2_hi = vget_high_f32(a2);
        float32x2_t a3_lo = vget_low_f32(a3), a3_hi = vget_high_f32(a3);
        
        c_row0 = vmlaq_lane_f32(c_row0, bt0, a0_lo, 0);
        c_row0 = vmlaq_lane_f32(c_row0, bt1, a0_lo, 1);
        c_row0 = vmlaq_lane_f32(c_row0, bt2, a0_hi, 0);
        c_row0 = vmlaq_lane_f32(c_row0, bt3, a0_hi, 1);
        
        c_row1 = vmlaq_lane_f32(c_row1, bt0, a1_lo, 0);
        c_row1 = vmlaq_lane_f32(c_row1, bt1, a1_lo, 1);
        c_row1 = vmlaq_lane_f32(c_row1, bt2, a1_hi, 0);
        c_row1 = vmlaq_lane_f32(c_row1, bt3, a1_hi, 1);
        
        c_row2 = vmlaq_lane_f32(c_row2, bt0, a2_lo, 0);
        c_row2 = vmlaq_lane_f32(c_row2, bt1, a2_lo, 1);
        c_row2 = vmlaq_lane_f32(c_row2, bt2, a2_hi, 0);
        c_row2 = vmlaq_lane_f32(c_row2, bt3, a2_hi, 1);
        
        c_row3 = vmlaq_lane_f32(c_row3, bt0, a3_lo, 0);
        c_row3 = vmlaq_lane_f32(c_row3, bt1, a3_lo, 1);
        c_row3 = vmlaq_lane_f32(c_row3, bt2, a3_hi, 0);
        c_row3 = vmlaq_lane_f32(c_row3, bt3, a3_hi, 1);
    }
    
    /* Remainder */
    for (; k < K; k++) {
        float a0k = a_row0[k], a1k = a_row1[k], a2k = a_row2[k], a3k = a_row3[k];
        float32x4_t b_col = {bt_row0[k], bt_row1[k], bt_row2[k], bt_row3[k]};
        
        c_row0 = vmlaq_n_f32(c_row0, b_col, a0k);
        c_row1 = vmlaq_n_f32(c_row1, b_col, a1k);
        c_row2 = vmlaq_n_f32(c_row2, b_col, a2k);
        c_row3 = vmlaq_n_f32(c_row3, b_col, a3k);
    }
    
    vst1q_f32(C, c_row0);
    vst1q_f32(C + ldc, c_row1);
    vst1q_f32(C + 2 * ldc, c_row2);
    vst1q_f32(C + 3 * ldc, c_row3);
}

/* ============================================================================
 * Tiled matrix multiply for a single tile of C
 * ============================================================================
 * 
 * Computes C[i0:i0+Ti][j0:j0+Tj] += A[i0:i0+Ti][k0:k0+Tk] × B[k0:k0+Tk][j0:j0+Tj]
 * 
 * With B transposed to BT:
 * C[i0:i0+Ti][j0:j0+Tj] += A[i0:i0+Ti][k0:k0+Tk] × BT[j0:j0+Tj][k0:k0+Tk]^T
 */

static void matmul_tile(
    const float *A,     /* Full matrix A */
    const float *BT,    /* Full transposed B */
    float *C,           /* Full matrix C */
    int n,              /* Matrix dimension */
    int i0, int j0,     /* Top-left corner of C tile */
    int Ti, int Tj,     /* Tile dimensions for C */
    int k0, int Tk      /* K range to process */
) {
    /* Process 4×4 micro-tiles within this tile */
    for (int i = i0; i < i0 + Ti; i += 4) {
        int i_end = (i + 4 <= i0 + Ti) ? 4 : (i0 + Ti - i);
        
        for (int j = j0; j < j0 + Tj; j += 4) {
            int j_end = (j + 4 <= j0 + Tj) ? 4 : (j0 + Tj - j);
            
            if (i_end == 4 && j_end == 4) {
                /* Full 4×4 micro-kernel */
                kernel_4x4_neon(
                    A + i * n + k0,     /* A[i][k0] */
                    BT + j * n + k0,    /* BT[j][k0] */
                    C + i * n + j,      /* C[i][j] */
                    n, n, n,            /* leading dimensions */
                    Tk                  /* K tile size */
                );
            } else {
                /* Scalar fallback for edge tiles */
                for (int ii = i; ii < i + i_end; ii++) {
                    for (int jj = j; jj < j + j_end; jj++) {
                        float sum = C[ii * n + jj];
                        for (int kk = k0; kk < k0 + Tk; kk++) {
                            sum += A[ii * n + kk] * BT[jj * n + kk];
                        }
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

/* ============================================================================
 * NEON Single-Threaded Implementation with Tiling
 * ============================================================================ */

void matmul_neon_single(const float *A, const float *B, float *C, int n) {
    float *BT = (float *)malloc(n * n * sizeof(float));
    if (!BT) return;
    
    transpose_matrix(B, BT, n);
    memset(C, 0, n * n * sizeof(float));
    
    /* 
     * Tiled loop order: i-tiles, j-tiles, k-tiles
     * 
     * The k-loop is innermost so that for each (i,j) tile of C,
     * we accumulate contributions from all k-tiles before moving on.
     * This keeps the C tile in cache while streaming through A and BT.
     */
    const int T = TILE_SIZE;
    
    for (int i0 = 0; i0 < n; i0 += T) {
        int Ti = (i0 + T <= n) ? T : (n - i0);
        
        for (int j0 = 0; j0 < n; j0 += T) {
            int Tj = (j0 + T <= n) ? T : (n - j0);
            
            for (int k0 = 0; k0 < n; k0 += T) {
                int Tk = (k0 + T <= n) ? T : (n - k0);
                
                matmul_tile(A, BT, C, n, i0, j0, Ti, Tj, k0, Tk);
            }
        }
    }
    
    free(BT);
}

/* ============================================================================
 * NEON + OpenMP Multi-Threaded Implementation with Tiling
 * ============================================================================ */

void matmul_neon_omp(const float *A, const float *B, float *C, int n) {
    float *BT = (float *)malloc(n * n * sizeof(float));
    if (!BT) return;
    
    transpose_matrix(B, BT, n);
    memset(C, 0, n * n * sizeof(float));
    
    const int T = TILE_SIZE;
    
    /* Parallelize over i-tiles (rows of C) */
    #pragma omp parallel for schedule(static)
    for (int i0 = 0; i0 < n; i0 += T) {
        int Ti = (i0 + T <= n) ? T : (n - i0);
        
        for (int j0 = 0; j0 < n; j0 += T) {
            int Tj = (j0 + T <= n) ? T : (n - j0);
            
            for (int k0 = 0; k0 < n; k0 += T) {
                int Tk = (k0 + T <= n) ? T : (n - k0);
                
                matmul_tile(A, BT, C, n, i0, j0, Ti, Tj, k0, Tk);
            }
        }
    }
    
    free(BT);
}
