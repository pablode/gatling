add_library(
  imgio STATIC
  include/imgio.h
  include/img.h
  include/error_codes.h
  src/imgio.c
  src/jpeg.c
  src/jpeg.h
  src/png.c
  src/png.h
  src/mmap.h
  src/mmap.c
)

set_target_properties(
  imgio
  PROPERTIES
    LINKER_LANGUAGE C
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
    C_EXTENSIONS OFF
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
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
    turbojpeg-static
)
