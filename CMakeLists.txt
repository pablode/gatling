cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

include(CMakeDependentOption)

project(gatling VERSION 0.4.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

foreach(CONFIG_TYPE ${CMAKE_CONFIGURATION_TYPES})
  string(TOUPPER ${CONFIG_TYPE} CONFIG_TYPE)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${OUTPUT_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${OUTPUT_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_TYPE} ${OUTPUT_DIR})
endforeach()

option(GTL_BUILD_HDGATLING "Build the gatling hydra render delegate." ON)
option(GTL_VERBOSE "Enable verbose logging." OFF)

if(${GTL_BUILD_HDGATLING})
  find_package(USD REQUIRED HINTS ${USD_ROOT} NAMES pxr)
  find_package(MaterialX REQUIRED HINTS ${USD_ROOT})
endif()
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
