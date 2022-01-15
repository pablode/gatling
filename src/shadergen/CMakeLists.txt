add_library(
  shadergen STATIC
  include/shadergen.h
  src/shadergen.cpp
  src/IShaderCompiler.h
  src/IShaderCompiler.cpp
  src/MtlxMdlCodeGen.h
  src/MtlxMdlCodeGen.cpp
  src/MdlHlslCodeGen.h
  src/MdlHlslCodeGen.cpp
  src/MdlLogger.h
  src/MdlLogger.cpp
  src/MdlMaterialCompiler.h
  src/MdlMaterialCompiler.cpp
  src/MdlNeurayLoader.h
  src/MdlNeurayLoader.cpp
  src/MdlRuntime.h
  src/MdlRuntime.cpp
)

if(${GATLING_USE_DXC})
  target_sources(shadergen PRIVATE src/DxcShaderCompiler.h src/DxcShaderCompiler.cpp)
else()
  target_sources(shadergen PRIVATE src/GlslangShaderCompiler.h src/GlslangShaderCompiler.cpp)
endif()

set_target_properties(
  shadergen
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
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
)

if(${GATLING_USE_DXC})
  target_link_libraries(shadergen PRIVATE DXC)
  target_compile_definitions(shadergen PRIVATE GATLING_USE_DXC)
else()
  target_link_libraries(shadergen PRIVATE shaderc shaderc_util)
endif()

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
