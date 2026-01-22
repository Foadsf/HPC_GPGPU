@ cpu_matmul_neon.s
@ ARM NEON 4×4 Single-Precision Matrix Multiplication
@ C = A × B where A, B, C are 4×4 float matrices
@
@ Calling convention (AAPCS):
@   r0 = pointer to A (4×4 floats, row-major)
@   r1 = pointer to B (4×4 floats, row-major)
@   r2 = pointer to C (4×4 floats, output)
@
@ NEON registers used:
@   q0-q3:   Rows of matrix A
@   q4-q7:   Columns of matrix B (transposed during load)
@   q8-q11:  Rows of result matrix C
@   q12-q15: Temporary

.global cpu_matmul_4x4_neon
.text
.align 4

.fpu neon
.arch armv7-a

cpu_matmul_4x4_neon:
    @ Preserve callee-saved registers
    vpush   {q4-q7}
    
    @ =========================================================================
    @ Load matrix A (4 rows × 4 columns = 16 floats)
    @ A is row-major: A[row][col] = A[row*4 + col]
    @ =========================================================================
    vld1.32 {q0}, [r0]!         @ q0 = A row 0: [A00, A01, A02, A03]
    vld1.32 {q1}, [r0]!         @ q1 = A row 1: [A10, A11, A12, A13]
    vld1.32 {q2}, [r0]!         @ q2 = A row 2: [A20, A21, A22, A23]
    vld1.32 {q3}, [r0]          @ q3 = A row 3: [A30, A31, A32, A33]
    
    @ =========================================================================
    @ Load matrix B columns (need to transpose for efficient multiply)
    @ B is row-major, but we need columns for dot products
    @ We'll load B rows then transpose using VTRN/VZIP
    @ =========================================================================
    vld1.32 {q4}, [r1]!         @ q4 = B row 0: [B00, B01, B02, B03]
    vld1.32 {q5}, [r1]!         @ q5 = B row 1: [B10, B11, B12, B13]
    vld1.32 {q6}, [r1]!         @ q6 = B row 2: [B20, B21, B22, B23]
    vld1.32 {q7}, [r1]          @ q7 = B row 3: [B30, B31, B32, B33]
    
    @ Transpose B: convert rows to columns
    @ After transpose:
    @   q4 = [B00, B10, B20, B30] = B column 0
    @   q5 = [B01, B11, B21, B31] = B column 1
    @   q6 = [B02, B12, B22, B32] = B column 2
    @   q7 = [B03, B13, B23, B33] = B column 3
    
    @ 4×4 transpose using VTRN (transpose 2×2 blocks)
    vtrn.32 q4, q5              @ Swap pairs: q4=[B00,B10,B02,B12], q5=[B01,B11,B03,B13]
    vtrn.32 q6, q7              @ Swap pairs: q6=[B20,B30,B22,B32], q7=[B21,B31,B23,B33]
    vswp    d9, d12             @ Swap d9 (q4 high) with d12 (q6 low)
    vswp    d11, d14            @ Swap d11 (q5 high) with d14 (q7 low)
    
    @ =========================================================================
    @ Compute C = A × B using dot products
    @ C[i][j] = sum(A[i][k] * B[k][j]) for k=0..3
    @
    @ With NEON, we compute one row of C at a time:
    @ C_row_i = [dot(A_row_i, B_col_0), dot(A_row_i, B_col_1), ...]
    @ =========================================================================
    
    @ --- Row 0 of C ---
    vmul.f32  q8, q0, q4        @ q8 = A_row0 * B_col0 (element-wise)
    vmul.f32  q12, q0, q5       @ q12 = A_row0 * B_col1
    vmul.f32  q13, q0, q6       @ q13 = A_row0 * B_col2
    vmul.f32  q14, q0, q7       @ q14 = A_row0 * B_col3
    
    @ Horizontal add to get dot products
    @ vpadd adds adjacent pairs: [a,b,c,d] -> [a+b, c+d]
    vpadd.f32 d16, d16, d17     @ d16 = [q8[0]+q8[1], q8[2]+q8[3]]
    vpadd.f32 d24, d24, d25     @ d24 = [q12[0]+q12[1], q12[2]+q12[3]]
    vpadd.f32 d26, d26, d27     @ d26 = [q13[0]+q13[1], q13[2]+q13[3]]
    vpadd.f32 d28, d28, d29     @ d28 = [q14[0]+q14[1], q14[2]+q14[3]]
    
    vpadd.f32 d16, d16, d24     @ d16 = [C00, C01]
    vpadd.f32 d17, d26, d28     @ d17 = [C02, C03]
    @ q8 now contains C row 0: [C00, C01, C02, C03]
    
    @ --- Row 1 of C ---
    vmul.f32  q9, q1, q4
    vmul.f32  q12, q1, q5
    vmul.f32  q13, q1, q6
    vmul.f32  q14, q1, q7
    
    vpadd.f32 d18, d18, d19
    vpadd.f32 d24, d24, d25
    vpadd.f32 d26, d26, d27
    vpadd.f32 d28, d28, d29
    
    vpadd.f32 d18, d18, d24
    vpadd.f32 d19, d26, d28
    @ q9 now contains C row 1
    
    @ --- Row 2 of C ---
    vmul.f32  q10, q2, q4
    vmul.f32  q12, q2, q5
    vmul.f32  q13, q2, q6
    vmul.f32  q14, q2, q7
    
    vpadd.f32 d20, d20, d21
    vpadd.f32 d24, d24, d25
    vpadd.f32 d26, d26, d27
    vpadd.f32 d28, d28, d29
    
    vpadd.f32 d20, d20, d24
    vpadd.f32 d21, d26, d28
    @ q10 now contains C row 2
    
    @ --- Row 3 of C ---
    vmul.f32  q11, q3, q4
    vmul.f32  q12, q3, q5
    vmul.f32  q13, q3, q6
    vmul.f32  q14, q3, q7
    
    vpadd.f32 d22, d22, d23
    vpadd.f32 d24, d24, d25
    vpadd.f32 d26, d26, d27
    vpadd.f32 d28, d28, d29
    
    vpadd.f32 d22, d22, d24
    vpadd.f32 d23, d26, d28
    @ q11 now contains C row 3
    
    @ =========================================================================
    @ Store result matrix C
    @ =========================================================================
    vst1.32 {q8},  [r2]!        @ Store C row 0
    vst1.32 {q9},  [r2]!        @ Store C row 1
    vst1.32 {q10}, [r2]!        @ Store C row 2
    vst1.32 {q11}, [r2]         @ Store C row 3
    
    @ Restore callee-saved registers
    vpop    {q4-q7}
    
    bx      lr                   @ Return

.section .note.GNU-stack,"",%progbits
