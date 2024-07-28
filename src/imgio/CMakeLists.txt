add_library(
  imgio STATIC
  gtl/imgio/ErrorCodes.h
  gtl/imgio/Image.h
  gtl/imgio/Imgio.h
  impl/Imgio.cpp
  impl/ExrDecoder.h
  impl/ExrDecoder.cpp
  impl/HdrDecoder.h
  impl/HdrDecoder.cpp
  impl/JpegDecoder.h
  impl/JpegDecoder.cpp
  impl/PngDecoder.h
  impl/PngDecoder.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/imgio
    impl
)

target_link_libraries(
  imgio
  PRIVATE
    spng
    turbojpeg-static
    OpenEXR::OpenEXR
    stb
)

target_compile_options(
  imgio
  PRIVATE
    ${GTL_COMPILE_OPTIONS}
)
