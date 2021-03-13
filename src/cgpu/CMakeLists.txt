add_library(
  cgpu STATIC
  include/cgpu.h
  src/cgpu.c
  src/handle_store.c
  src/handle_store.h
  src/resource_store.c
  src/resource_store.h
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
)

if(ENABLE_IPO)
  set_target_properties(cgpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
