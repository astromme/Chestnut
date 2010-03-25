project(cuda_examples)

find_package(CUDA REQUIRED)
set(CUDA_BUILD_EMULATION ON)

cuda_add_executable(example-helloworld helloworld.cu)
cuda_add_executable(example-array-2d 2d_array_example.cu)
cuda_add_executable(fluid fluid.cu)
