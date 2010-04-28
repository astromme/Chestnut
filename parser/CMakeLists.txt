project(ChestnutParse)
cmake_minimum_required(VERSION 2.8)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(chestnut_parser chestnut.y ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp)
FLEX_TARGET(chestnut_scanner chestnut.l ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
#ADD_FLEX_BISON_DEPENDENCY(chestnut_scanner chestnut_parser)

find_package(Boost COMPONENTS regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(parseutils_SOURCES 
    SymbolTable.cpp
    ParseUtils.cpp
    ParseUtils_Functions.cpp
    ParseUtils_InputOutput.cpp
    ParseUtils_Timer.cpp
    ParseUtils_Variables.cpp
    ParseUtils_Foreach.cpp
    ParseUtils_HelperFunctions.cpp
    FileUtils.cpp
)

message("Flex Found: ${chestnut_scanner} ${FLEX_chestnut_scanner_OUTPUTS} ${FLEX_chestnut_scanner_DEFINED}")

add_executable(chestnut_parse
    ${BISON_chestnut_parser_OUTPUTS}
    ${FLEX_chestnut_scanner_OUTPUTS}
    ${parseutils_SOURCES}
    #${Boost_LIBRARIES}
  )

target_link_libraries(chestnut_parse fl ${Boost_LIBRARIES})
