cmake_minimum_required(VERSION 3.10)
project(GPGPU_GLES)

set(CMAKE_CXX_STANDARD 17)

find_package(glfw3 CONFIG REQUIRED)

add_executable(gpgpu_mm main.cpp)
target_link_libraries(gpgpu_mm PRIVATE glfw GLESv2)

# Copy shaders to build dir
add_custom_command(TARGET gpgpu_mm POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/vertex.glsl"
        "${CMAKE_CURRENT_SOURCE_DIR}/fragment.glsl"
        $<TARGET_FILE_DIR:gpgpu_mm>
)
