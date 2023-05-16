add_library(
  ggpu STATIC
  include/linearDataStore.h
  include/resizableBuffer.h
  include/stager.h
  include/syncBuffer.h
  src/linearDataStore.cpp
  src/resizableBuffer.cpp
  src/stager.cpp
  src/syncBuffer.cpp
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
