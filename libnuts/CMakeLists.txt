project(libnuts)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
include_directories(${libnuts_SOURCE_DIR} ${libnuts_BINARY_DIR})
include_directories("/usr/local/cuda/include" ${CMAKE_CURRENT_SOURCE_DIR})
#include_directories("../thrust" ${CMAKE_CURRENT_SOURCE_DIR})

set(CUDA_BUILD_EMULATION OFF)


set(nuts_sources
    DisplayWindow.cpp
    DeviceData.cu
    )


set(nuts_headers
    DisplayWindow.h
    DeviceData.h
    ColorKernels.h
    )

set(nuts_ui
    )

set(nuts_resource_files
    )

qt4_wrap_cpp(nuts_mocs ${nuts_headers})
qt4_wrap_ui(nuts_ui ${nuts_ui})
qt4_add_resources(nuts_resources ${nuts_resource_files})

cuda_add_library(nuts ${nuts_sources} ${nuts_mocs} ${nuts_ui} ${nuts_resources})
target_link_libraries(nuts ${QT_LIBRARIES})



