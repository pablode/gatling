add_library(
  ggpu STATIC
  include/resizableBuffer.h
  src/resizableBuffer.cpp
  include/stager.h
  src/stager.cpp
)

target_include_directories(
  ggpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  ggpu
  PRIVATE
    gb
    cgpu
)

set_target_properties(
  ggpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)

if(ENABLE_IPO)
  set_target_properties(ggpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
