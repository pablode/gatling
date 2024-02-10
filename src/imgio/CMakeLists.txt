add_library(
  imgio STATIC
  include/ErrorCodes.h
  include/Image.h
  include/Imgio.h
  src/Imgio.cpp
  src/ExrDecoder.h
  src/ExrDecoder.cpp
  src/HdrDecoder.h
  src/HdrDecoder.cpp
  src/JpegDecoder.h
  src/JpegDecoder.cpp
  src/PngDecoder.h
  src/PngDecoder.cpp
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
