# CUB DeviceReduce::Sum Benchmark Tool
# Modern CMake build system for CUDA/CUB reduction

cmake_minimum_required(VERSION 3.18)
project(cub_reduce LANGUAGES CXX CUDA)

# C++17 and CUDA C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Architecture option (RTX 3090 = SM_86 by default)
# Override with: cmake -DCMAKE_CUDA_ARCHITECTURES=75 for RTX 2080 Ti
# or cmake -DCMAKE_CUDA_ARCHITECTURES="75;86" for multiple architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Main executable
add_executable(cub_reduce main.cu)

# C++17 features
target_compile_features(cub_reduce PUBLIC cxx_std_17)

# Link against NVIDIA ML library for energy measurement
target_link_libraries(cub_reduce nvidia-ml)

# CUDA-specific compiler options for performance
target_compile_options(cub_reduce PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3                    # Maximum optimization
        --use_fast_math        # Fast math operations
        --restrict             # Enable __restrict__ keyword
        -lineinfo              # Generate line info for profiling (optional)
    >
)

# CXX compiler options
target_compile_options(cub_reduce PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3                    # Maximum optimization
        -march=native          # Optimize for host CPU (optional)
    >
)

# CUB is included in CCCL (CUDA Cooperative Groups and Collective)
# Modern CUDA toolkits (11.0+) include CUB automatically
# No additional find_package or external dependencies needed

# Optional: Enable debug symbols in Debug builds
target_compile_options(cub_reduce PRIVATE
    $<$<CONFIG:Debug>:
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G>
        $<$<COMPILE_LANGUAGE:CXX>:-g>
    >
)

# Link with CUDA runtime (automatic with CUDA language enabled)
# find_package(CUDAToolkit REQUIRED) is optional - only needed for specific components

# Print configuration info
message(STATUS "CUB Reduce Tool Configuration:")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CMAKE_CUDA_STANDARD: ${CMAKE_CUDA_STANDARD}")

# Optional: Installation target
install(TARGETS cub_reduce DESTINATION bin)
