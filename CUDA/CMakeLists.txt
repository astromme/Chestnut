project(cuda_examples)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)

set(CUDA_BUILD_EMULATION OFF)
include_directories("/usr/local/cuda/include" ${CMAKE_CURRENT_SOURCE_DIR})
include(${QT_USE_FILE})

set(fluid_sources
    fluid.cpp
    FluidView.cpp
    )

set(fluid_headers
    FluidView.h
    )

qt4_wrap_cpp(fluid_mocs ${fluid_headers})

cuda_add_executable(example-helloworld helloworld.cu)
cuda_add_executable(example-array-2d 2d_array_example.cu)
cuda_add_executable(example-array-2d-aligned 2d_array_aligned_example.cu)
cuda_add_executable(example-map map_example.cu)
add_executable(example-map-cpp map_example_cpp.cpp)
#cuda_add_executable(example-matrix-operations matrix-operations.cu)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(fluid ${fluid_sources} ${fluid_mocs})
target_link_libraries(fluid ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY})
