#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

// Tile Height in the K dimension
#define TILE_K 16

__kernel void matmul_tiled(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int N)
{
    // Coordinate Setup
    const int tx = get_local_id(0);       // 0..7
    const int bx = get_group_id(0);       // Block X
    const int by = get_global_id(1);      // Global Y (Row)
    
    // We want to compute C[by, global_x_vec]
    // The "Global X Vector" index for this thread is:
    const int global_vec_idx = bx * BLOCK_SIZE + tx;
    
    // Local Memory for B
    // We store a tile of B: 16 rows (TILE_K) x 8 vectors (BLOCK_SIZE) width
    // Size = 128 float16 vectors
    __local float16 tile_B[TILE_K * BLOCK_SIZE];

    float16 sum = 0.0f;

    // Loop over tiles
    for (int k = 0; k < N; k += TILE_K) {
        
        // --- COOPERATIVE LOAD ---
        // We have 8 threads. We need to load 16 rows of B.
        // Each thread loads 2 rows.
        
        // Row 1
        int k_offset_1 = tx; // 0..7
        if (k + k_offset_1 < N) {
            // Read B[k+tx, global_vec_idx] ? No.
            // We need to load the B tile corresponding to THIS group's columns.
            // B is accessed as B[row, col].
            // We need B[k + k_offset, (bx * BLOCK_SIZE * 16) ... ]
            
            // This is actually very tricky because B is Row-Major.
            // Loading "Columns" of B is not contiguous in memory.
            // Vectors must be loaded from contiguous addresses: B[row, col...col+15]
            
            // This means one vload16 gets us 16 columns for 1 row.
            // Since our thread needs exactly that vector, we can just load it directly.
            
            // Let's have each thread load the vectors IT needs for the next TILE_K rows?
            // No, that's just prefetching.
            
            // Let's try: Each thread loads 2 rows of the tile for the *whole block*?
            // No, width is distributed across threads.
            
            // Correct approach for Tiling Row-Major B with Vectors:
            // Every thread loads its OWN column-strip for TILE_K rows into Local Mem.
            // Thread 'tx' is responsible for B column-vectors 'global_vec_idx' across 'TILE_K' rows.
            
            for (int i = 0; i < TILE_K; i++) {
                // Thread 'tx' loads B[k+i, global_vec_idx] into Local Memory
                // Local Index: [i * BLOCK_SIZE + tx]
                if (k + i < N) {
                    tile_B[i * BLOCK_SIZE + tx] = vload16(0, &B[(k + i) * N + global_vec_idx * 16]);
                } else {
                    tile_B[i * BLOCK_SIZE + tx] = 0.0f;
                }
            }
        }
        
        // Wait for all loads to finish
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // --- COMPUTE ---
        for (int i = 0; i < TILE_K; i++) {
             if (k + i < N) {
                 float a_val = A[by * N + (k + i)];
                 // Read from fast Local Memory
                 float16 b_val = tile_B[i * BLOCK_SIZE + tx];
                 sum += a_val * b_val;
             }
        }
        
        // Wait before overwriting Local Memory in next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    vstore16(sum, 0, &C[by * N + global_vec_idx * 16]);
}
