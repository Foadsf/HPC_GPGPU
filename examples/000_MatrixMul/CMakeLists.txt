cmake_minimum_required(VERSION 3.15)
project(GPGPU_MatrixMul)

find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(gpgpu_mm main.cpp)

target_link_libraries(gpgpu_mm PRIVATE glfw glad::glad)
set_property(TARGET gpgpu_mm PROPERTY CXX_STANDARD 17)

add_custom_command(TARGET gpgpu_mm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/compute.glsl"
        $<TARGET_FILE_DIR:gpgpu_mm>
)
