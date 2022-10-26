add_library(
  shadergen STATIC
  include/ShaderGen.h
  src/ShaderGen.cpp
  src/GlslangShaderCompiler.h
  src/GlslangShaderCompiler.cpp
  src/MdlGlslCodeGen.h
  src/MdlGlslCodeGen.cpp
  src/MdlLogger.h
  src/MdlLogger.cpp
  src/MdlMaterialCompiler.h
  src/MdlMaterialCompiler.cpp
  src/MdlNeurayLoader.h
  src/MdlNeurayLoader.cpp
  src/MdlRuntime.h
  src/MdlRuntime.cpp
  src/MtlxDocumentPatcher.h
  src/MtlxDocumentPatcher.cpp
  src/MtlxMdlCodeGen.h
  src/MtlxMdlCodeGen.cpp
)

set_target_properties(
  shadergen
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(
  shadergen
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
    ${MDL_INCLUDE_DIR}
)

target_link_libraries(
  shadergen
  PRIVATE
    MaterialXCore
    MaterialXFormat
    MaterialXGenMdl
    glslang
    SPIRV
)

install(
  FILES
    ${MDL_SHARED_LIB}
  DESTINATION
    "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT
    hdGatling
)

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  DESTINATION
    "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT
    hdGatling
)

install(
  DIRECTORY
    "${CMAKE_CURRENT_SOURCE_DIR}/mdl/"
  DESTINATION
    "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/mdl"
  COMPONENT
    hdGatling
)
