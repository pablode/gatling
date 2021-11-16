add_library(
  imgio STATIC
  include/imgio.h
  include/img.h
  include/error_codes.h
  src/imgio.c
  src/png.c
  src/png.h
  src/mmap.h
  src/mmap.c
)

target_include_directories(
  imgio
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  imgio
  PRIVATE
    spng
)
