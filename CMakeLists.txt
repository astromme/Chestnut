project(Chestnut)
cmake_minimum_required(VERSION 2.8)

#set( CUDA_NVCC_FLAGS "--compiler-bindir=/usr/bin/gcc-4.3")

add_subdirectory(libnuts)
add_subdirectory(parser)
add_subdirectory(CUDA)
add_subdirectory(sequential)
add_subdirectory(thrust_examples)
add_subdirectory(gui)

if(BUILD_CHESTNUT)
  message("Building CHESTNUT file chestnut.cu")
  add_subdirectory(chestnut)
else()
  message("NOT building CHESTNUT file chestnut.cu. To build it, pass -DBUILD_CHESTNUT=true to the 'cmake' command")
endif(BUILD_CHESTNUT)
