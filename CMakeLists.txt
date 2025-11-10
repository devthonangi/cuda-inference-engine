cmake_minimum_required(VERSION 3.10)
project(CUDA_Inference_Engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(include)

add_executable(cuda_inference_engine
    src/main.cpp
    src/inference_engine.cpp
    src/cuda_kernels.cu
)

target_link_libraries(cuda_inference_engine
    ${CUDA_LIBRARIES}
    nvinfer
    nvonnxparser
    cudart
    cublas
)
