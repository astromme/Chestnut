project(Chestnut)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
include_directories(${Chestnut_SOURCE_DIR} ${Chestnut_BINARY_DIR})

set(chestnut_datablocks
    array1d.cpp
    array2d.cpp
    array3d.cpp
    )

set(chestnut_operators
    plus.cpp
    minus.cpp
    )

set(chestnut_functions
    map.cpp
    reduce.cpp
    sort.cpp
    )

set(chestnut_sources
    chestnut.cpp
    function.cpp
    datablock.cpp
    operator.cpp
    value.cpp
    ${chestnut_functions}
    ${chestnut_operators}
    ${chestnut_datablocks}
    )


set(chestnut_headers
    )

set(chestnut_ui
    mainwindow.ui
    )

qt4_wrap_cpp(chestnut_mocs ${chestnut_headers})
qt4_wrap_ui(chestnut_ui ${chestnut_ui})

add_executable(chestnut ${chestnut_sources} ${chestnut_mocs} ${chestnut_ui})
target_link_libraries(chestnut ${QT_LIBRARIES})


