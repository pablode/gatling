cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(CMakeDependentOption)

project(gatling VERSION 0.3.0)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

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

option(GATLING_USE_EMBREE "Use Embree for BVH building." ON)
cmake_dependent_option(GATLING_USE_DXC "Use DXC instead of glslang" OFF WIN32 OFF)

find_package(Vulkan REQUIRED)
find_package(USD REQUIRED HINTS ${USD_ROOT} NAMES pxr)
find_package(MDL REQUIRED)

if(${GATLING_USE_DXC})
  find_package(DXC REQUIRED)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ENABLE_IPO)

include(CheckLibraryExists)
check_library_exists(m exp2f "" C_MATH_LIBRARY_EXISTS)
if(C_MATH_LIBRARY_EXISTS)
  set(C_MATH_LIBRARY "m")
endif()

if(WIN32)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_USE_MATH_DEFINES)
endif()

add_subdirectory(extern)
add_subdirectory(src)
