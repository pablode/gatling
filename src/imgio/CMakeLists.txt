add_library(
  imgio STATIC
  include/imgio.h
  include/img.h
  include/error_codes.h
  src/imgio.cpp
  src/exr.h
  src/exr.cpp
  src/hdr.h
  src/hdr.cpp
  src/jpeg.h
  src/jpeg.cpp
  src/png.h
  src/png.cpp
)

set_target_properties(
  imgio
  PROPERTIES
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
    OpenEXR::OpenEXR
    stb
)
