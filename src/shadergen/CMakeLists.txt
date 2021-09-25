add_library(
  shadergen STATIC
  include/shadergen.h
  src/shadergen.cpp
  src/IShaderCompiler.h
  src/IShaderCompiler.cpp
  src/GlslangShaderCompiler.h
  src/GlslangShaderCompiler.cpp
)

set_target_properties(
  shadergen
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(
  shadergen
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
)

target_link_libraries(
  shadergen
  PRIVATE
    shaderc
    shaderc_util
)

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  DESTINATION
    "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT
    hdGatling
)
