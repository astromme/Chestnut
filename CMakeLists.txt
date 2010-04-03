project(CS87-FinalProject)
cmake_minimum_required(VERSION 2.8)

add_subdirectory(parser)
add_subdirectory(CUDA)

if(BUILD_CUDALANG)
  message("Building CUDALANG file cudalang.cpp")
  add_subdirectory(cudalang)
else()
  message("NOT building CUDALANG file cudalang.cpp. To build it, pass -DBUILD_CUDALANG=true to the 'cmake' command")
endif(BUILD_CUDALANG)
