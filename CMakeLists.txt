cmake_minimum_required(VERSION 3.20)
project(vulkan-nn-lib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Vulkan
# ------------------------------------------------------------
find_package(Vulkan QUIET)
if(NOT Vulkan_FOUND)
    message(WARNING
        "Vulkan SDK not found. Install it with:\n"
        "  sudo apt install libvulkan-dev vulkan-tools mesa-vulkan-drivers\n"
        "or download the LunarG SDK from https://vulkan.lunarg.com/\n"
        "Build will continue but GPU targets will not be usable.")
endif()

# ------------------------------------------------------------
# Shader compilation (handled inside shaders/CMakeLists.txt)
# ------------------------------------------------------------
add_subdirectory(shaders)

# ------------------------------------------------------------
# Library sources
# ------------------------------------------------------------
set(LIB_SOURCES
    src/vk_context.cpp
    src/vk_buffer.cpp
    src/vk_pipeline.cpp
    src/inference.cpp
)

add_library(vulkan_nn STATIC ${LIB_SOURCES})
target_include_directories(vulkan_nn PUBLIC src)

if(Vulkan_FOUND)
    target_link_libraries(vulkan_nn PUBLIC Vulkan::Vulkan)
    target_compile_definitions(vulkan_nn PUBLIC VULKAN_AVAILABLE)
endif()

# ------------------------------------------------------------
# Main executable
# ------------------------------------------------------------
add_executable(vulkan_nn_main src/main.cpp)
target_link_libraries(vulkan_nn_main PRIVATE vulkan_nn)

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------
enable_testing()

add_executable(test_relu tests/test_relu.cpp)
target_link_libraries(test_relu PRIVATE vulkan_nn)
target_compile_definitions(test_relu PRIVATE
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
add_test(NAME test_relu COMMAND test_relu)

add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE vulkan_nn)
target_compile_definitions(test_matmul PRIVATE
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
add_test(NAME test_matmul COMMAND test_matmul)

add_executable(test_buffer tests/test_buffer.cpp)
target_link_libraries(test_buffer PRIVATE vulkan_nn)
add_test(NAME test_buffer COMMAND test_buffer)

# ------------------------------------------------------------
# Benchmark
# ------------------------------------------------------------
add_executable(bench_cpu_vs_gpu benchmark/bench_cpu_vs_gpu.cpp)
target_link_libraries(bench_cpu_vs_gpu PRIVATE vulkan_nn)
