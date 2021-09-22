add_library(
  shadergen STATIC
  include/shadergen.h
  src/shadergen.cpp
)

set_target_properties(
  shadergen
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(
  shadergen
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)
