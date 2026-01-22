// Vectorized matrix multiplication for VC4CL (Raspberry Pi GPU)
// Each work-item computes 16 output elements of C (one float16 vector)
// This utilizes the QPU's native SIMD-16 architecture

__kernel void matmul_simple(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    // Row index (y) remains the same (0 to N)
    const int row = get_global_id(1);
    
    // Column index (x) now represents a BLOCK of 16 elements
    const int col_vec_idx = get_global_id(0);
    
    // Bounds check
    if (row >= N || col_vec_idx >= N/16) return;
    
    // Accumulator for 16 separate dot products
    float16 sum = 0.0f;
    
    // Calculate the actual starting column index for this vector
    const int col_start = col_vec_idx * 16;
    
    // Loop over the shared dimension K
    for (int k = 0; k < N; k++) {
        // Load 1 scalar from A and broadcast it to all 16 lanes
        // A is accessed as scalar: A[row, k]
        float a_val = A[row * N + k];
        
        // Load 16 contiguous floats from B
        // B is accessed as vector: B[k, col_start ... col_start+15]
        float16 b_vec = vload16(0, &B[k * N + col_start]);
        
        // Fused Multiply-Add (vectorized)
        // This computes 16 partial sums in parallel
        sum += a_val * b_vec;
    }
    
    // Store the final 16 results to C
    vstore16(sum, 0, &C[row * N + col_start]);
}
