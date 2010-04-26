project(Chestnut)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
include_directories(${Chestnut_SOURCE_DIR} ${Chestnut_BINARY_DIR})


set(chestnut_operators
    standardoperation.cpp
    )

set(chestnut_functions
    map.cpp
    reduce.cpp
    sort.cpp
    print.cpp
    write.cpp
    )

set(chestnut_datatypes
    datablock.cpp
    value.cpp
    )

set(chestnut_sources
    chestnut.cpp
    mainwindow.cpp
    scene.cpp
    palettemodel.cpp
    palettedelegate.cpp
    object.cpp
    function.cpp
    operation.cpp
    data.cpp
    source.cpp
    sink.cpp
    connection.cpp
    drawingutils.cpp
    datautils.cpp
    ${chestnut_functions}
    ${chestnut_operators}
    ${chestnut_datatypes}
    )


set(chestnut_headers
    sink.h
    source.h
    object.h
    datablock.h
    value.h
    mainwindow.h
    )

set(chestnut_ui
    mainwindow.ui
    datablock.ui
    value.ui
    output.ui
    )

set(chestnut_resource_files
    icons.qrc
    )

qt4_wrap_cpp(chestnut_mocs ${chestnut_headers})
qt4_wrap_ui(chestnut_ui ${chestnut_ui})
qt4_add_resources(chestnut_resources ${chestnut_resource_files})

add_executable(chestnut ${chestnut_sources} ${chestnut_mocs} ${chestnut_ui} ${chestnut_resources})
target_link_libraries(chestnut ${QT_LIBRARIES})


