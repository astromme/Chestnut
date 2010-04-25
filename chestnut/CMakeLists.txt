project(chestnut)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
#find_package(Qt4 REQUIRED)

set(CUDA_BUILD_EMULATION OFF)
include_directories("/usr/local/cuda/include" ${CMAKE_CURRENT_SOURCE_DIR})
#include(${QT_USE_FILE})

set(chestnut_program_sources
    chestnut.cu
    chestnut.cpp
    ParseLibs.cpp
    ParseLibs.inl
    )

# set(fluid_headers
#     FluidView.h
#     )

# qt4_wrap_cpp(fluid_mocs ${fluid_headers})

cuda_add_executable(chestnut_example ${chestnut_program_sources})

set(CMAKE_BUILD_TYPE RelWithDebInfo)

# add_executable(fluid ${fluid_sources} ${fluid_mocs})
# target_link_libraries(fluid ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY})
