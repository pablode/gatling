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

# Required since library is linked into hdGatling DSO
set_target_properties(gml PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_IPO)
  set_target_properties(gml PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
