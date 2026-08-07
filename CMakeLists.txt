cmake_minimum_required(VERSION 3.18)
project(InferenceEngine LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

set(CMAKE_CUDA_ARCHITECTURES 89)

set(KERNEL_SOURCES
    src/kernels/tensor_utilities.cu
    src/kernels/softmax.cu
    src/kernels/norm.cu
    src/kernels/rope.cu
    src/kernels/embedding.cu
    src/kernels/elementwise.cu
    src/kernels/attention.cu   
    src/kernels/linear.cu
    src/kernels/mm_t.cu
    src/kernels/buffer.cu
    src/kernels/atten_kernels.cu
    
)


add_library(kernels STATIC ${KERNEL_SOURCES})


target_include_directories(kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "native"
)


add_executable(inference_engine
    cJSON/cJSON.c
    src/weight_loading.c
    src/tokenizer.cpp
    src/load_tokenizer.cpp
    src/models/qwen_model.cpp
    src/main.cpp              
    # src/benchmark.cpp               
)

target_include_directories(inference_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cJSON
)

target_link_libraries(inference_engine PRIVATE kernels)

set_target_properties(inference_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "native"
)