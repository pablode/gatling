add_library(
  cgpu
  src/cgpu.c
  src/handle_store.c
  src/resource_store.c
)

target_include_directories(
  cgpu
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE src
)

target_link_libraries(
  cgpu PRIVATE
  volk
)
