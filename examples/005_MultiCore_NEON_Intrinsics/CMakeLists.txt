cmake_minimum_required(VERSION 3.10)
project(005_MultiCore_NEON_Intrinsics C)

# Require C11 standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Cortex-A53 specific optimization flags
# -mcpu=cortex-a53: Target the specific CPU in Raspberry Pi 3B
# -mfpu=neon-vfpv4: Enable NEON with VFPv4 (fused multiply-add)
# -mfloat-abi=hard: Use hardware floating point ABI
# -O3: Maximum optimization level
# -ffast-math: Allow aggressive FP optimizations (slightly less precise)
# -funroll-loops: Unroll loops for better pipelining
set(CORTEX_A53_FLAGS "-mcpu=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(OPTIMIZATION_FLAGS "-O3 -ffast-math -funroll-loops")

# Combine all C flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CORTEX_A53_FLAGS} ${OPTIMIZATION_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

# Source files
set(SOURCES
    main.c
    matmul_neon_omp.c
)

# Create executable
add_executable(matmul_neon_omp ${SOURCES})

# Link libraries
target_link_libraries(matmul_neon_omp 
    OpenMP::OpenMP_C
    m  # Math library
)

# Compiler warnings (separate from optimization to keep output clean)
target_compile_options(matmul_neon_omp PRIVATE -Wall -Wextra -Wno-unused-parameter)

# Print configuration summary
message(STATUS "")
message(STATUS "=== 005_MultiCore_NEON_Intrinsics Build Configuration ===")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "OpenMP Version: ${OpenMP_C_VERSION}")
message(STATUS "")
