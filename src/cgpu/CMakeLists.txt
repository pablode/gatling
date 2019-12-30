add_library(
  cgpu
  cgpu.c
  cgpu.h
  handle_store.c
  handle_store.h
  resource_store.c
  resource_store.h
)

target_include_directories(
  cgpu PRIVATE
  ${GATLING_SOURCE_DIR}
)

target_compile_options(
  cgpu PRIVATE
)

target_link_libraries(
  cgpu PRIVATE
  volk
)
