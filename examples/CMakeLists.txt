project(examples)

find_package(Chestnut)

include_directories(${CHESTNUT_INCLUDE_DIRS})

chestnut_add_executable(3d 3d.chestnut)
chestnut_add_executable(diffusion diffusion.chestnut)
#chestnut_add_executable(file_io file_io.chestnut)
chestnut_add_executable(gradient gradient.chestnut)
chestnut_add_executable(game_of_life game_of_life.chestnut)
chestnut_add_executable(heatflow heatflow.chestnut)
chestnut_add_executable(io_test io_test.chestnut)
#chestnut_add_executable(julia julia.chestnut)
chestnut_add_executable(mandelbrot mandelbrot.chestnut)
chestnut_add_executable(matrix_multiply matrix_multiply.chestnut)
chestnut_add_executable(minimal minimal.chestnut)
#chestnut_add_executable(people_simulation people_simulation.chestnut)
chestnut_add_executable(random random.chestnut)
chestnut_add_executable(ripple ripple.chestnut)
chestnut_add_executable(small_game_of_life small_game_of_life.chestnut)
#chestnut_add_executable(sort sort.chestnut)
#chestnut_add_executable(structures structures.chestnut)
#chestnut_add_executable(vector_addition vector_addition.chestnut)
