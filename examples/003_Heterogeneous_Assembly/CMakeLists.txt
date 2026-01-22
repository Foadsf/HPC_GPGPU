cmake_minimum_required(VERSION 3.10)
project(003_Heterogeneous_Assembly C ASM)

# Find vc4asm
find_program(VC4ASM_EXE vc4asm REQUIRED)

# CPU Assembly Source
set(CPU_ASM_SRC cpu_ops.s)

# GPU Assembly Source
set(QPU_ASM_SRC qpu_kernel.qasm)
set(QPU_HEX ${CMAKE_CURRENT_BINARY_DIR}/qpu_kernel.hex)

# Custom Command: Assemble .qasm -> .hex (C hex fragment)
add_custom_command(
    OUTPUT ${QPU_HEX}
    COMMAND ${VC4ASM_EXE} -C ${QPU_HEX} -i vc4.qinc ${CMAKE_CURRENT_SOURCE_DIR}/${QPU_ASM_SRC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${QPU_ASM_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Assembling QPU Kernel with vc4asm"
)

# Main Executable
add_executable(rpi_asm_host main.c ${CPU_ASM_SRC})

# Make the hex file a dependency
add_custom_target(qpu_kernel_hex DEPENDS ${QPU_HEX})
add_dependencies(rpi_asm_host qpu_kernel_hex)

# Add the binary directory to include path so main.c finds qpu_kernel.hex
target_include_directories(rpi_asm_host PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
