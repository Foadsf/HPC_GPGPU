cmake_minimum_required(VERSION 3.10)
project(GPGPU_GLES)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 CONFIG REQUIRED)

# Link against system GLESv2
add_executable(gpgpu_mm main.cpp)
target_link_libraries(gpgpu_mm PRIVATE glfw GLESv2)
