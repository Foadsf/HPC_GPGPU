cmake_minimum_required(VERSION 3.10)
project(006_Zero_Copy_Shared_Memory C)

# Require C11 standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Target: Raspberry Pi 3B (Cortex-A53, ARMv8-A in 32-bit mode)
# These flags ensure optimal performance for the memory benchmarks
#
# -mcpu=cortex-a53: Target the specific CPU
# -mfpu=neon-vfpv4: Enable NEON SIMD for optimized memory operations
# -mfloat-abi=hard: Hardware floating point
# -O3: Maximum optimization
# -funroll-loops: Unroll loops for better memory throughput
set(CORTEX_A53_FLAGS "-mcpu=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard")
set(OPTIMIZATION_FLAGS "-O3 -funroll-loops")

# Combine flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CORTEX_A53_FLAGS} ${OPTIMIZATION_FLAGS}")

# Source files
set(SOURCES
    main.c
    mailbox.c
)

# Header files (for IDE integration)
set(HEADERS
    mailbox.h
)

# Create executable
add_executable(zero_copy_bench ${SOURCES} ${HEADERS})

# Link libraries
# - rt: Real-time library for clock_gettime (CLOCK_MONOTONIC)
target_link_libraries(zero_copy_bench
    rt
)

# Compiler warnings
target_compile_options(zero_copy_bench PRIVATE 
    -Wall 
    -Wextra 
    -Wno-unused-parameter
    -Wno-unused-variable
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== 006_Zero_Copy_Shared_Memory Build Configuration ===")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C Flags: ${CMAKE_C_FLAGS}")
message(STATUS "Target: Raspberry Pi 3B (Cortex-A53, BCM2837)")
message(STATUS "")
message(STATUS "Note: This benchmark requires root privileges to access")
message(STATUS "      /dev/vcio (mailbox) and /dev/mem (physical memory)")
message(STATUS "")

# Installation (optional)
install(TARGETS zero_copy_bench DESTINATION bin)

# Custom target for running as root
add_custom_target(run
    COMMAND sudo ${CMAKE_CURRENT_BINARY_DIR}/zero_copy_bench
    DEPENDS zero_copy_bench
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running Zero-Copy benchmark as root..."
)

# Custom target for running with different sizes
add_custom_target(run-small
    COMMAND sudo ${CMAKE_CURRENT_BINARY_DIR}/zero_copy_bench 16
    DEPENDS zero_copy_bench
    COMMENT "Running benchmark with 16 MB data..."
)

add_custom_target(run-large
    COMMAND sudo ${CMAKE_CURRENT_BINARY_DIR}/zero_copy_bench 128
    DEPENDS zero_copy_bench
    COMMENT "Running benchmark with 128 MB data..."
)
