add_library(
  cgpu STATIC
  include/cgpu.h
  src/cgpu.c
  src/handle_store.c
  src/handle_store.h
  src/resource_store.c
  src/resource_store.h
  src/shader_reflection.c
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
    volk
    vma-c
    spirv-reflect-static
)

set_target_properties(
  cgpu
  PROPERTIES
    LINKER_LANGUAGE C
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)

if(ENABLE_IPO)
  set_target_properties(cgpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
