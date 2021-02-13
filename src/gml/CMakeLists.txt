add_library(
  gml STATIC
  include/gml.h
  src/gml.c
)

target_include_directories(
  gml
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  gml
  PRIVATE
    ${C_MATH_LIBRARY}
)
