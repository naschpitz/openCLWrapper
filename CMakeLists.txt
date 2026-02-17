cmake_minimum_required(VERSION 3.14)

project(OpenCLWrapper LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)

# Find OpenCL - consumers will need this
find_package(OpenCL REQUIRED)

add_library(OpenCLWrapper STATIC
  OCLW_Core.hpp OCLW_CU.hpp OCLW_Kernel.hpp
  OCLW_Core.cpp OCLW_CU.cpp
  LICENSE.md
)

# PUBLIC: Qt and OpenCL are needed by consumers that use OpenCLWrapper
target_link_libraries(OpenCLWrapper
  PUBLIC Qt${QT_VERSION_MAJOR}::Core
  PUBLIC OpenCL::OpenCL
)

# PUBLIC include directory so consumers can #include <OCLW_Core.hpp>
target_include_directories(OpenCLWrapper PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(OpenCLWrapper PRIVATE OPENCLWRAPPER_LIBRARY)
