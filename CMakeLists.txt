cmake_minimum_required(VERSION 3.18)
project(BN254_CUDA LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Auto-detect GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try to detect GPU compute capability
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(GPU_COMPUTE_CAP)
        string(REPLACE "." "" GPU_ARCH ${GPU_COMPUTE_CAP})
        set(CMAKE_CUDA_ARCHITECTURES ${GPU_ARCH})
        message(STATUS "Detected GPU compute capability: ${GPU_COMPUTE_CAP} (arch ${GPU_ARCH})")
    else()
        # Default to common architectures if detection fails
        set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
        message(STATUS "Could not detect GPU, using default architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(FIELD_OPS_SOURCES
    src/field_ops.cu
    src/bn254_api.cpp
)

# Create static library for field operations
add_library(bn254_field_ops STATIC ${FIELD_OPS_SOURCES})
set_target_properties(bn254_field_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(bn254_field_ops CUDA::cudart)

# Test executables
add_executable(test_basic tests/test_basic.cu)
target_link_libraries(test_basic bn254_field_ops CUDA::cudart)
set_target_properties(test_basic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_vectors tests/test_vectors.cu)
target_link_libraries(test_vectors bn254_field_ops CUDA::cudart)
set_target_properties(test_vectors PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_performance tests/test_performance.cu)
target_link_libraries(test_performance bn254_field_ops CUDA::cudart)
set_target_properties(test_performance PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Install targets
install(TARGETS bn254_field_ops
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES
    include/bn254_types.cuh
    include/field_ops.cuh
    DESTINATION include
)

install(TARGETS test_basic test_vectors test_performance
    RUNTIME DESTINATION bin
)

# Custom targets for running tests
add_custom_target(run_test_basic
    COMMAND test_basic
    DEPENDS test_basic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running basic functionality tests"
)

add_custom_target(run_test_vectors
    COMMAND test_vectors
    DEPENDS test_vectors
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running vector operation tests"
)

add_custom_target(run_test_performance
    COMMAND test_performance
    DEPENDS test_performance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running performance benchmarks"
)

add_custom_target(run_all_tests
    COMMAND test_basic && test_vectors && test_performance
    DEPENDS test_basic test_vectors test_performance
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all tests"
)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  Project: ${PROJECT_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
