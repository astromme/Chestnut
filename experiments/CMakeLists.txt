project(experiments)

find_package(CUDA)
find_package(Chestnut)

include_directories(${CHESTNUT_INCLUDE_DIRS})

find_path(CUDA_CUT_INCLUDE_DIR
  cutil.h
  PATHS ${CUDA_SDK_SEARCH_PATH}
  PATH_SUFFIXES "common/inc"
  DOC "Location of cutil.h"
  NO_DEFAULT_PATH
  )

### find cutil library
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

chestnut_add_executable(game_of_life_chestnut game_of_life.chestnut)
cuda_add_executable(game_of_life_cuda game_of_life.cu)
#add_executable(game_of_life game_of_life.cpp)

chestnut_add_executable(mandelbrot_chestnut mandelbrot.chestnut)
cuda_add_executable(mandelbrot_cuda mandelbrot.cu)
add_executable(mandelbrot_sequential mandelbrot.cpp)

chestnut_add_executable(matrix_multiply_chestnut matrix_multiply.chestnut)
cuda_add_executable(matrix_multiply_cuda matrix_multiply.cu)
add_executable(matrix_multiply_sequential matrix_multiply.cpp)
target_link_libraries(matrix_multiply_cuda ${CUDA_CUT_LIBRARIES})

chestnut_add_executable(heatflow_chestnut heatflow.chestnut)
cuda_add_executable(heatflow_cuda heatflow.cu)
add_executable(heatflow_sequential heatflow.cpp)



chestnut_add_executable(bigGOL_chestnut big_gol.chestnut)
chestnut_add_executable(bigheat_chestnut bigheatflow.chestnut)
chestnut_add_executable(bigmandelbrot_chestnut bigmandelbrot.chestnut)
chestnut_add_executable(bigmatrixmult_chestnut bigmm.chestnut) 
target_link_libraries(bigmatrixmult_chestnut ${CUDA_CUT_LIBRARIES})
cuda_add_executable(bigmatrixmult_cuda bigmm.cu) 
cuda_add_executable(bigheat_cuda bigheatflow.cu) 
cuda_add_executable(bigmanelbrot bigmandelbrot.cu)
