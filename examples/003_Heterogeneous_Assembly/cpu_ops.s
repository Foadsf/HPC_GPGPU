@ cpu_ops.s
@ ARMv7 (32-bit ARM) Assembly for Raspberry Pi 3B running 32-bit OS
@ Signature: int cpu_add_asm(int a, int b);

.global cpu_add_asm
.text
.align 2

cpu_add_asm:
    @ In ARMv7 (32-bit):
    @ r0 = first argument (a)
    @ r1 = second argument (b)
    @ Result is returned in r0

    add r0, r0, r1    @ r0 = r0 + r1
    bx  lr            @ Return to caller (branch to link register)

.section .note.GNU-stack,"",%progbits
