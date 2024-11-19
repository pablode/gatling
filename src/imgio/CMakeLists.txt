set(IMGIO_SRCS
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
  impl/TiffDecoder.h
  impl/TiffDecoder.cpp
)

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PRIVATE
      gtl/imgio
      impl
  )

  target_link_libraries(
    ${TARGET}
    PRIVATE
      spng
      turbojpeg-static
      OpenEXR::OpenEXR
      stb # for HDR
      tiff tiffxx
  )
endfunction()

add_library(imgio STATIC ${IMGIO_SRCS})
target_include_directories(
  imgio
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
  imgio
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
configure_target(imgio)

add_executable(imgio_test ${IMGIO_SRCS} impl/main.cpp)
target_link_libraries(imgio_test PRIVATE gt doctest)
target_compile_definitions(imgio_test
  PRIVATE
    IMGIO_TESTENV_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testenv"
)
configure_target(imgio_test)
