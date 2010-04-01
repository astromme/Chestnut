project(TestMap)
cmake_minimum_required(VERSION 2.8)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(testmap_parser testmap.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(testmap_scanner testmap.l  ${CMAKE_CURRENT_BIANRY_DIR}/lexer.cpp)
ADD_FLEX_BISON_DEPENDENCY(testmap_scanner testmap_parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_executable(testmap
    ${BISON_testmap_parser_OUTPUTS}
    ${FLEX_testmap_scanner_OUTPUTS}
  )
