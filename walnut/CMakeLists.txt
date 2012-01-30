project(libwalnut)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)
SET(WALNUT_LIBRARY)

find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)
find_package(Qt4 REQUIRED)
find_package(OpenGL REQUIRED)
include(${QT_USE_FILE})
include_directories(${OPENGL_INCLUDE_DIRS})
include_directories(${libwalnut_SOURCE_DIR} ${libwalnut_BINARY_DIR})

if (APPLE)
  # Directory used on OSX with the default cuda install
  include_directories("/usr/local/cuda/include")
endif (APPLE)

set(CUDA_BUILD_EMULATION OFF)


set(walnut_sources
    UtilityFunctors.cu
    FunctionIterator.cpp
    HostFunctions.cpp
    DisplayWindow.cpp
    Array.cu
    Windows.cu
    )

set(walnut_headers
    UtilityFunctors.h
    FunctionIterator.h
    HostFunctions.h
    walnut_global.h
    gl_extensions.h
    DisplayWindow.h
    ColorKernels.h
    ArrayAllocator.h
    Array.h
    Sizes.h
    Points.h
    Windows.h
    Color.h
    )

set(walnut_templates
    ArrayAllocator.cpp
    )

set(walnut_ui
    )

set(walnut_resource_files
    )

qt4_wrap_cpp(walnut_mocs ${walnut_headers})
qt4_wrap_ui(walnut_ui ${walnut_ui})
qt4_add_resources(walnut_resources ${walnut_resource_files})

cuda_add_library(walnut SHARED ${walnut_sources} ${walnut_headers} ${walnut_mocs} ${walnut_ui} ${walnut_resources})
target_link_libraries(walnut ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY} ${OPENGL_LIBRARIES})

install(TARGETS walnut DESTINATION lib)
install(FILES ${walnut_headers} ${walnut_templates} DESTINATION "include/walnut")
