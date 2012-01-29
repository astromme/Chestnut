project(Chestnut)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
set(CMAKE_INCLUDE_PATH ${CMAKE_SOURCE_DIR})
set(CHESTNUT_COMPILER_DIR ${CMAKE_SOURCE_DIR}/scripts)
set(CHESTNUT_USE_INTERNAL_LIBRARY true)

add_subdirectory(walnut)
add_subdirectory(gui)
add_subdirectory(examples)
add_subdirectory(experiments)
