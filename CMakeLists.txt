project(CS87-FinalProject)
cmake_minimum_required(VERSION 2.8)

add_subdirectory(parser)
add_subdirectory(CUDA)
add_subdirectory(thrust_examples)

if(BUILD_CHESTNUT)
  message("Building CHESTNUT file cudalang.cu")
  add_subdirectory(chestnut)
else()
  message("NOT building CHESTNUT file cudalang.cu. To build it, pass -DBUILD_CHESTNUT=true to the 'cmake' command")
endif(BUILD_CHESTNUT)
