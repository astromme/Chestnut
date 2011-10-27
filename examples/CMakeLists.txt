project(cuda_examples)
cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
find_package(Qt4 REQUIRED)
#find_package(Walnut REQUIRED)
set(WALNUT_LIBRARIES "-L ${CMAKE_CURRENT_BINARY_DIR}" "-lwalnut")

set(CUDA_BUILD_EMULATION OFF)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include(${QT_USE_FILE})

#set(SOURCES mandelbrot.chestnut)
#
#set(generated_cu_files)
#foreach(_file ${SOURCES})
#  set(file_output "${_file}.cu")
#  message(${file_output})
#  add_custom_command(
#    OUTPUT ${file_output} 
#    COMMAND chestnut-compiler
#    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/${_file} -o ${file_output}
#    DEPENDS ${_file}
#  ) 
#  list(APPEND generated_cu_files ${file_output})
#
#endforeach()
#add_custom_target(do_chestnut_compile DEPENDS ${generated_cu_files})
#set_property(SOURCE mandelbrot.chestnut.cu APPEND PROPERTY OBJECT_DEPENDS mandelbrot.chestnut)

#cuda_add_executable(mandelbrot mandelbrot.cu)
#target_link_libraries(mandelbrot ${WALNUT_LIBRARIES} ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY})
#add_dependencies(mandelbrot ${generated_cu_files})


macro(CHESTNUT_ADD_EXECUTABLE target SOURCE)
add_custom_command(
  OUTPUT  ${SOURCE}.cu
  COMMAND chestnut-compiler
  ARGS    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE} -o ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cu
  DEPENDS ${SOURCE}
)
set_property(SOURCE ${SOURCE}.cu APPEND PROPERTY OBJECT_DEPENDS ${SOURCE}.chestnut)

cuda_add_executable(${target} ${CMAKE_CURRENT_BINARY_DIR}/${SOURCE}.cu)
target_link_libraries(${target} ${WALNUT_LIBRARIES} ${QT_LIBRARIES} ${QT_QTOPENGL_LIBRARY})
endmacro(CHESTNUT_ADD_EXECUTABLE target SOURCE)

#chestnut_add_executable(file_io file_io.chestnut)
chestnut_add_executable(gradient gradient.chestnut)
chestnut_add_executable(heatflow heatflow.chestnut)
chestnut_add_executable(io_test io_test.chestnut)
#chestnut_add_executable(julia julia.chestnut)
chestnut_add_executable(mandelbrot mandelbrot.chestnut)
chestnut_add_executable(matrix_multiply matrix_multiply.chestnut)
chestnut_add_executable(minimal minimal.chestnut)
#chestnut_add_executable(people_simulation people_simulation.chestnut)
chestnut_add_executable(ripple ripple.chestnut)
chestnut_add_executable(small_game_of_life small_game_of_life.chestnut)
#chestnut_add_executable(sort sort.chestnut)
#chestnut_add_executable(structures structures.chestnut)
#chestnut_add_executable(vector_addition vector_addition.chestnut)
