cmake_minimum_required(VERSION 3.10)
project(004_Heterogeneous_MatrixMul C ASM)

# Enable NEON for ARM assembly
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpu=neon -mfloat-abi=hard")
set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon -mfloat-abi=hard")

# Find vc4asm
find_program(VC4ASM_EXE vc4asm REQUIRED)
message(STATUS "Found vc4asm: ${VC4ASM_EXE}")

# Source files
set(CPU_ASM_SRC cpu_matmul_neon.s)
set(QPU_ASM_SRC qpu_matmul.qasm)
set(QPU_HEX ${CMAKE_CURRENT_BINARY_DIR}/qpu_matmul.hex)

# Custom command: Assemble QPU code
add_custom_command(
    OUTPUT ${QPU_HEX}
    COMMAND ${VC4ASM_EXE} -V -C ${QPU_HEX} -i vc4.qinc ${CMAKE_CURRENT_SOURCE_DIR}/${QPU_ASM_SRC}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${QPU_ASM_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Assembling QPU kernel with vc4asm"
)

# Create custom target for QPU hex
add_custom_target(qpu_kernel_hex DEPENDS ${QPU_HEX})

# Main executable
add_executable(matmul_asm main.c ${CPU_ASM_SRC})

# Ensure QPU hex is built before main
add_dependencies(matmul_asm qpu_kernel_hex)

# Include directory for generated hex file
target_include_directories(matmul_asm PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# Link math library
target_link_libraries(matmul_asm m)

# Compiler warnings
target_compile_options(matmul_asm PRIVATE -Wall -Wextra)
