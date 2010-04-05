project(TestMap)
cmake_minimum_required(VERSION 2.8)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(testmap_parser testmap.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(testmap_scanner testmap.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
#ADD_FLEX_BISON_DEPENDENCY(testmap_scanner testmap_parser)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(parseutils_SOURCES 
    SymbolTable.cpp
    ParseUtils.cpp
)

message("Flex Found: ${testmap_scanner} ${FLEX_testmap_scanner_OUTPUTS} ${FLEX_testmap_scanner_DEFINED}")

add_executable(testmap
    ${BISON_testmap_parser_OUTPUTS}
    ${FLEX_testmap_scanner_OUTPUTS}
    ${parseutils_SOURCES}
  )

target_link_libraries(testmap fl)
