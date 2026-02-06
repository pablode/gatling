cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

include(CMakeDependentOption)

project(gatling VERSION 0.5.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GTL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GTL_OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GTL_OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GTL_OUTPUT_DIR}")

foreach(CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${CONFIG_TYPE} CONFIG_TYPE)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${GTL_OUTPUT_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${GTL_OUTPUT_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${GTL_OUTPUT_DIR})
endforeach()

set(GTL_TEST_OUTPUT_DIR "${PROJECT_SOURCE_DIR}/TEST")
set(GTL_HDGATLING_RESOURCES_DIR "./hdGatling/resources")
set(GTL_OIDN_WEIGHTS_FILE "rt_hdr_small.tza")

option(GTL_VERBOSE "Enable verbose logging." OFF)

find_package(OpenGL REQUIRED) # Required due to USD bug #3309
find_package(MaterialX REQUIRED HINTS ${USD_ROOT})
find_package(USD REQUIRED HINTS ${USD_ROOT} NAMES pxr)
find_package(MDL REQUIRED)
find_package(OpenMP)

include(CheckLibraryExists)
check_library_exists(m exp2f "" C_MATH_LIBRARY_EXISTS)
if(C_MATH_LIBRARY_EXISTS)
  set(C_MATH_LIBRARY "m")
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

include(cmake/BuildConfig.cmake)

add_subdirectory(extern)
add_subdirectory(src)
