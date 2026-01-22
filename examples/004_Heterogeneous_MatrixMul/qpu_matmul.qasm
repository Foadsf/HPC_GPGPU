# qpu_matmul.qasm
# VideoCore IV QPU - 16-Element Vector Multiply
# 
# Computes C[i] = A[i] * B[i] for i=0..15 (element-wise product)
# This demonstrates QPU SIMD computation with TMU reads and VPM/DMA writes.
#
# Uniforms (parameters from host):
#   uniform[0] = address of vector A (16 floats, contiguous)
#   uniform[1] = address of vector B (16 floats, contiguous)  
#   uniform[2] = address of output C (16 floats)
#   uniform[3] = number of elements (16)

# =============================================================================
# Register file allocation to avoid conflicts
# Rule: Can only read ONE reg from file A and ONE from file B per instruction
# Accumulators r0-r3 have no such restriction
# =============================================================================

.set ra_addr_a,     ra0     # Address of A (file A)
.set ra_addr_b,     ra1     # Address of B (file A)  
.set ra_addr_out,   ra2     # Address of output (file A)
.set ra_a_vec,      ra3     # A vector data (file A)

.set rb_b_vec,      rb0     # B vector data (file B)

# =============================================================================
# Load Uniforms
# =============================================================================

mov ra_addr_a, unif         # uniform[0]: address of A
mov ra_addr_b, unif         # uniform[1]: address of B
mov ra_addr_out, unif       # uniform[2]: address of C (output)
mov r0, unif                # uniform[3]: count (16) - into accumulator

# Get per-element index (0-15)
mov r1, elem_num            # r1 = SIMD lane number [0,1,2,...,15]

# =============================================================================
# Load A[0..15] via TMU
# =============================================================================
# Each SIMD lane loads its own element: lane i loads A[i]
# Address for lane i = base_addr + i * 4

shl r2, r1, 2               # r2 = elem_num * 4 (float offset in bytes)
add r2, ra_addr_a, r2       # r2 = &A[elem_num]

# Issue TMU request
mov tmu0_s, r2

# Wait for TMU (typically ~9 cycles, use NOPs or useful work)
nop; nop; nop; nop
nop; nop; nop; nop

# Receive TMU result
nop; ldtmu0                 # Signal: result ready in r4
mov ra_a_vec, r4            # ra_a_vec[i] = A[i]

# =============================================================================
# Load B[0..15] via TMU  
# =============================================================================

shl r2, r1, 2               # r2 = elem_num * 4
add r2, ra_addr_b, r2       # r2 = &B[elem_num]

mov tmu0_s, r2

nop; nop; nop; nop
nop; nop; nop; nop

nop; ldtmu0
mov rb_b_vec, r4            # rb_b_vec[i] = B[i]

# Need 1 instruction delay before reading rb_b_vec after writing it
nop

# =============================================================================
# Compute C = A * B (element-wise)
# =============================================================================
# ra_a_vec is in file A, rb_b_vec is in file B - no conflict!

fmul r0, ra_a_vec, rb_b_vec # r0[i] = A[i] * B[i], all 16 lanes parallel

# =============================================================================
# Store result via VPM + DMA (proven method from 003)
# =============================================================================

# VPM Write Setup: horizontal, 32-bit words, Y=0
# Format: [31:30]=0 (QPU write), [13:12]=1 (horiz), [9:8]=2 (32-bit), [7:0]=Y,X
ldi vw_setup, 0x00001a00

# Write the 16 results to VPM
mov vpm, r0

# VPM DMA Store Setup: write 1 row of 16 words from VPM to memory  
# Format: [31:30]=2 (DMA store), [22:16]=0 (units-1), [15:13]=0 (depth-1)
#         [12:11]=0 (horiz), [10]=0 (32-bit), [6:3]=Y, [2:0]=X/4
ldi vw_setup, 0x80904000

# Trigger DMA by writing destination address
mov vw_addr, ra_addr_out

# Wait for DMA completion
mov r3, vw_wait

# =============================================================================
# Done - signal host
# =============================================================================

mov interrupt, 1

thrend
nop
nop
