project(thrust_examples)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)

set(CUDA_BUILD_EMULATION OFF)
include_directories("/usr/local/cuda/include" ${CMAKE_CURRENT_SOURCE_DIR})
#include_directories("../thrust" ${CMAKE_CURRENT_SOURCE_DIR})
include(${QT_USE_FILE})

cuda_add_executable(thrust-sum sum.cu)
cuda_add_executable(thrust-minmax minmax.cu)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
