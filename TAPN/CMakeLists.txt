cmake_minimum_required(VERSION 3.28)
project(TAPN LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)

# Shared CUDA code in a static library
add_library(petri STATIC
    petri/Arcs.cu
    petri/Invariant.cu
    petri/Net.cu
    petri/Place.cu
    observer/Observer.cu
    petri/Transition.cu
)

# Required for separable compilation to work across targets
set_target_properties(petri PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    POSITION_INDEPENDENT_CODE ON
)

# Euler executable
add_executable(euler
    petri/euler.cu
)
target_link_libraries(euler PRIVATE petri)
set_target_properties(euler PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# Fireflies executable
add_executable(fireflies
    petri/fireflies.cu
)
target_link_libraries(fireflies PRIVATE petri)
set_target_properties(fireflies PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)

# Parser executable (includes rapidxml)
add_executable(parser
    parser/parser.cu
)
target_include_directories(parser PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../rapidxml)
target_link_libraries(parser PRIVATE petri)
set_target_properties(parser PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)
