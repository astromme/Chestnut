project(sequential)
cmake_minimum_required(VERSION 2.6)

include_directories("/usr/include/c++/4.5.0") # for Ryan's iostream

set(sequential_sources
    mapreduce.cpp
    )

add_executable(mapreduce ${sequential_sources})
add_executable(mandelbrot mandelbrot.cpp)
add_executable(matrix_multiply matrix_multiply.cpp)

find_package(Boost COMPONENTS program_options REQUIRED)
add_executable(game_of_life GameOfLife.cpp)
target_link_libraries(game_of_life ${Boost_PROGRAM_OPTIONS_LIBRARY})

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
