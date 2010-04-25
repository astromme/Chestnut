project(sequential)
cmake_minimum_required(VERSION 2.6)

include_directories("/usr/include/c++/4.5.0") # for Ryan's iostream

set(sequential_sources
    mapreduce.cpp
    )

add_executable(mapreduce ${sequential_sources})

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
