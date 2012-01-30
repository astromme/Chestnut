find_package(CUDA REQUIRED)
find_package(Thrust REQUIRED)
find_package(Qt4 REQUIRED)

find_path(WALNUT_INCLUDE_DIR
    HINTS /usr/include /usr/local/include
    NAMES walnut/walnut_global.h
    DOC "Walnut headers")

if (CHESTNUT_USE_INTERNAL_LIBRARY)
    set(WALNUT_LIBRARY walnut)
else()
    set(WALNUT_NAMES ${WALNUT_NAMES} walnut)
    find_library(WALNUT_LIBRARY NAMES ${WALNUT_NAMES})
endif (CHESTNUT_USE_INTERNAL_LIBRARY)

if (WALNUT_INCLUDE_DIR AND WALNUT_LIBRARY)
    list(REMOVE_DUPLICATES WALNUT_INCLUDE_DIR)

    # Create the symlink to avoid cuda complaining about kernel launches from system files
    execute_process(COMMAND ${CMAKE_COMMAND} -E
        create_symlink
        ${WALNUT_INCLUDE_DIR}/walnut
        ${CMAKE_BINARY_DIR}/walnut)
else()
    message(FATAL_ERROR "Couldn't find the walnut library and headers")
endif (WALNUT_INCLUDE_DIR AND WALNUT_LIBRARY)

set(WALNUT_INCLUDE_DIRS ${WALNUT_INCLUDE_DIR})
set(CHESTNUT_INCLUDE_DIRS ${WALNUT_INCLUDE_DIR})

find_program(CHESTNUT_COMPILER 
             NAMES chestnut-compiler
             PATHS $ENV{CHESTNUT_COMPILER_DIR} ${CHESTNUT_COMPILER_DIR} /usr/bin /usr/local/bin
                   /usr/local/share/python
             DOC "The chestnut-compiler executable for the Chestnut installation to use")

if(CHESTNUT_COMPILER)
  message("Using Chestnut Compiler ${CHESTNUT_COMPILER}")
else()
  message(FATAL_ERROR "Failed to find the chestnut compiler. Try setting CHESTNUT_COMPILER_DIR")
endif(CHESTNUT_COMPILER)

# handle the QUIETLY and REQUIRED arguments and set WALNUT_FOUND to TRUE if 
# all listed variables are TRUE

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CHESTNUT DEFAULT_MSG WALNUT_LIBRARY WALNUT_INCLUDE_DIR)

MARK_AS_ADVANCED(WALNUT_INCLUDE_DIR WALNUT_LIBRARY)

#find_package(Walnut REQUIRED)
#set(WALNUT_LIBRARIES "-L ${CMAKE_CURRENT_BINARY_DIR}" "-lwalnut")

###### HACK TO FIND CUTIL Library ######
find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )

set(cuda_cutil_name cutil_x86_64)

find_library(CUDA_CUT_LIBRARY
  NAMES cutil ${cuda_cutil_name}
  PATHS ${CUDA_SDK_SEARCH_PATH}
  # The new version of the sdk shows up in common/lib, but the old one is in lib
  PATH_SUFFIXES "common/lib" "lib"
  DOC "Location of cutil library"
  NO_DEFAULT_PATH
  )
# Now search system paths
find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
mark_as_advanced(CUDA_CUT_LIBRARY)
set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
### end find cutil library

include_directories(${CUDA_CUT_INCLUDE_DIR})
###### END HACK ######

set(CUDA_BUILD_EMULATION OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include(${QT_USE_FILE})

macro(CHESTNUT_ADD_EXECUTABLE target SOURCE)
  include_directories(${WALNUT_INCLUDE_DIR})
  include_directories(${THRUST_INCLUDE_DIR})

  add_custom_command(
    OUTPUT  ${SOURCE}.cu
    COMMAND ${CHESTNUT_COMPILER}
    ARGS    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cu
    DEPENDS ${SOURCE}
  )
  set_property(SOURCE ${SOURCE}.cu APPEND PROPERTY OBJECT_DEPENDS ${SOURCE}.chestnut)

  cuda_add_executable(${target} ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cu)
  target_link_libraries(${target} ${WALNUT_LIBRARY} ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY} ${CUDA_CUT_LIBRARIES})
endmacro(CHESTNUT_ADD_EXECUTABLE target SOURCE)
