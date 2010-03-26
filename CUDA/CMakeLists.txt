project(cuda_examples)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)

set(CUDA_BUILD_EMULATION OFF)
add_definitions(QT_DEFINITIONS)

cuda_add_executable(example-helloworld helloworld.cu)
cuda_add_executable(example-array-2d 2d_array_example.cu)
cuda_add_executable(example-array-2d-aligned 2d_array_aligned_example.cu)
cuda_add_executable(fluid fluid.cu)

