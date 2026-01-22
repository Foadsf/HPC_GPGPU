# qpu_kernel.qasm
# Minimal QPU kernel: Write 0x1337 to memory address from uniform
# VideoCore IV QPU assembly for vc4asm

# Read uniform (memory address where we write)
mov r0, unif

# Load magic value
ldi r1, 0x1337

# Setup VPM for horizontal 32-bit write
# Bits: [31:30]=0 (write setup), [29:28]=0, [27:14]=stride, [13:12]=horiz, 
#       [11:10]=laned, [9:8]=size(32-bit=2), [7:4]=Y, [3:0]=X
ldi vw_setup, 0x00001a00

# Write value to VPM
mov vpm, r1

# Setup VPM DMA store (VCD)
# ID=2 (DMA store setup), units=1, depth=1, etc.
ldi vw_setup, 0x80904000

# Trigger DMA write to memory address in r0
mov vw_addr, r0

# Wait for VPM DMA to complete
# vw_wait is rb50 - reading it stalls until DMA complete
# Can't combine with small immediate, so use mov instead
mov r2, vw_wait

# Signal host interrupt
mov interrupt, 1

# Program end - thrend needs 2 instructions after it
thrend
nop
nop
