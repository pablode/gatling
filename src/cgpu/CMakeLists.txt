
add_library(
  cgpu
  cgpu.cpp
  cgpu.hpp
  handle_store.cpp
  handle_store.hpp
)

target_include_directories(
  cgpu PRIVATE
  ${GATLING_SOURCE_DIR}
)

target_link_libraries(
  cgpu PRIVATE
  volk
)
