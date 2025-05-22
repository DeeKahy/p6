cmake_minimum_required(VERSION 3.28)
project(TAPN LANGUAGES CXX CUDA)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_ARCHITECTURES 61 75 86 89)  # change depending on your GPU

# Add the executable with .cu files
add_executable( TAPN    petri/fireflies.cu
                        petri/Arcs.cu
                        petri/Invariant.cu
                        petri/Net.cu
                        petri/Place.cu
                        petri/Transition.cu
                        observer/Observer.cu)  

set_target_properties(TAPN PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
)