cmake_minimum_required(VERSION 3.10)
project(VC4CL_MatrixMul C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL
find_package(OpenCL REQUIRED)

message(STATUS "OpenCL found: ${OpenCL_FOUND}")
message(STATUS "OpenCL include: ${OpenCL_INCLUDE_DIRS}")
message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")

# Create executable
add_executable(vc4cl_mm main.cpp)

target_include_directories(vc4cl_mm PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(vc4cl_mm PRIVATE ${OpenCL_LIBRARIES} m)

target_compile_options(vc4cl_mm PRIVATE -Wall -Wextra)

# Copy kernel to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/matmul.cl
               ${CMAKE_CURRENT_BINARY_DIR}/matmul.cl COPYONLY)
