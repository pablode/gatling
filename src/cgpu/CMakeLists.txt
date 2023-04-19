add_library(
  cgpu STATIC
  include/cgpu.h
  src/cgpu.cpp
  src/handle_store.cpp
  src/handle_store.h
  src/resource_store.cpp
  src/resource_store.h
  src/shader_reflection.cpp
  src/shader_reflection.h
)

target_include_directories(
  cgpu
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  cgpu
  PRIVATE
    gb
    volk
    vma-c
    spirv-reflect-static
)

set_target_properties(
  cgpu
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)

if(ENABLE_IPO)
  set_target_properties(cgpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
