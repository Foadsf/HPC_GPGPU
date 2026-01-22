cmake_minimum_required(VERSION 3.10)
project(GPGPU_MatrixMul_RPi3B C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Try to find GLES2 and EGL via pkg-config
pkg_check_modules(GLES2 REQUIRED glesv2)
pkg_check_modules(EGL REQUIRED egl)
pkg_check_modules(GBM gbm)
pkg_check_modules(DRM libdrm)

# Create the executable
add_executable(gpgpu_mm main.cpp)

# Include directories
target_include_directories(gpgpu_mm PRIVATE
    ${GLES2_INCLUDE_DIRS}
    ${EGL_INCLUDE_DIRS}
    ${GBM_INCLUDE_DIRS}
    ${DRM_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(gpgpu_mm PRIVATE
    ${GLES2_LIBRARIES}
    ${EGL_LIBRARIES}
    ${GBM_LIBRARIES}
    ${DRM_LIBRARIES}
    m
)

# Compiler flags
target_compile_options(gpgpu_mm PRIVATE
    ${GLES2_CFLAGS_OTHER}
    ${EGL_CFLAGS_OTHER}
    -Wall -Wextra
)

# Copy shader files to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/vertex.glsl
               ${CMAKE_CURRENT_BINARY_DIR}/vertex.glsl COPYONLY)
# Note: fragment.glsl is generated dynamically in main.cpp for the specific matrix size

# Installation
install(TARGETS gpgpu_mm DESTINATION bin)
install(FILES vertex.glsl fragment.glsl DESTINATION share/gpgpu_mm)

message(STATUS "GLES2 libraries: ${GLES2_LIBRARIES}")
message(STATUS "EGL libraries: ${EGL_LIBRARIES}")
message(STATUS "GBM libraries: ${GBM_LIBRARIES}")
