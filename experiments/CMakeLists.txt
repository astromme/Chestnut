project(experiments)

find_package(CUDA)
find_package(Chestnut)

include_directories(${CHESTNUT_INCLUDE_DIRS})

chestnut_add_executable(game_of_life_chestnut game_of_life.chestnut)
cuda_add_executable(game_of_life_cuda game_of_life.cu)
#add_executable(game_of_life game_of_life.cpp)

chestnut_add_executable(mandelbrot_chestnut mandelbrot.chestnut)
cuda_add_executable(mandelbrot_cuda mandelbrot.cu)
add_executable(mandelbrot_sequential mandelbrot.cpp)

chestnut_add_executable(matrix_multiply_chestnut matrix_multiply.chestnut)
cuda_add_executable(matrix_multiply_cuda matrix_multiply.cu)
add_executable(matrix_multiply_sequential matrix_multiply.cpp)

chestnut_add_executable(heatflow_chestnut heatflow.chestnut)
cuda_add_executable(heatflow_cuda heatflow.cu)
add_executable(heatflow_sequential heatflow.cpp)



chestnut_add_executable(bigGOL_chestnut big_gol.chestnut)
chestnut_add_executable(bigheat_chestnut bigheatflow.chestnut)
chestnut_add_executable(bigmandelbrot_chestnut bigmandelbrot.chestnut)
chestnut_add_executable(bigmatrixmult_chestnut bigmm.chestnut) 
cuda_add_executable(bigmatrixmult_cuda bigmm.cu) 
cuda_add_executable(bigheat_cuda bigheatflow.cu) 
cuda_add_executable(bigmandelbrot bigmandelbrot.cu)
