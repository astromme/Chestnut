project(sequential)
cmake_minimum_required(VERSION 2.6)

set(sequential_sources
    mapreduce.cpp
    )

add_executable(mapreduce ${sequential_sources})

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
