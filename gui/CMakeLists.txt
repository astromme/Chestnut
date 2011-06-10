project(Chestnut_Designer)
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})
include_directories(${Chestnut_Designer_SOURCE_DIR} ${Chestnut_Designer_BINARY_DIR})


set(designer_operators
    standardoperation.cpp
    )

set(designer_functions
    map.cpp
    reduce.cpp
    sort.cpp
    print.cpp
    write.cpp
    )

set(designer_datatypes
    datablock.cpp
    value.cpp
    )

set(designer_sources
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
    sizes.cpp
    ${designer_functions}
    ${designer_operators}
    ${designer_datatypes}
    )


set(designer_headers
    sink.h
    source.h
    connection.h
    object.h
    datablock.h
    value.h
    mainwindow.h
    )

set(designer_ui
    mainwindow.ui
    datablock.ui
    value.ui
    output.ui
    runoutput.ui
    )

set(designer_resource_files
    icons.qrc
    )

qt4_wrap_cpp(designer_mocs ${designer_headers})
qt4_wrap_ui(designer_ui ${designer_ui})
qt4_add_resources(designer_resources ${designer_resource_files})

add_executable(chestnut-designer ${designer_sources} ${designer_mocs} ${designer_ui} ${designer_resources})
target_link_libraries(chestnut-designer ${QT_LIBRARIES})


