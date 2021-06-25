add_library(
  shadergen STATIC
  include/shadergen.h
  src/shadergen.cpp
  src/VkGlslSyntax.h
  src/VkGlslSyntax.cpp
)

set_target_properties(
  shadergen
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 11
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
    MaterialXGenShader
)
