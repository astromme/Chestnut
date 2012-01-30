project(Chestnut)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
set(CMAKE_INCLUDE_PATH ${CMAKE_SOURCE_DIR})
set(CHESTNUT_COMPILER_DIR ${CMAKE_SOURCE_DIR}/scripts)
set(CHESTNUT_USE_INTERNAL_LIBRARY true)

add_subdirectory(walnut)
add_subdirectory(gui)
add_subdirectory(chestnut)
add_subdirectory(examples)
add_subdirectory(experiments)
add_subdirectory(scripts)

install(FILES
              ${CMAKE_SOURCE_DIR}/cmake/modules/FindThrust.cmake
              ${CMAKE_SOURCE_DIR}/cmake/modules/FindChestnut.cmake
        DESTINATION
              ${CMAKE_ROOT}/Modules)

########### Add uninstall target ###############
CONFIGURE_FILE(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
  IMMEDIATE @ONLY)
ADD_CUSTOM_TARGET(uninstall
  "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")


###### Add ability to create a package #########
set(CPACK_GENERATOR "DEB")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Andrew Stromme")

include(CPack)
