add_library(
  gi STATIC
  include/gi.h
  src/gi.cpp
  src/assetReader.h
  src/assetReader.cpp
  src/mmap.h
  src/mmap.cpp
  src/texsys.h
  src/texsys.cpp
  src/turbo.h
  src/sg/ShaderGen.h
  src/sg/ShaderGen.cpp
  src/sg/GlslangShaderCompiler.h
  src/sg/GlslangShaderCompiler.cpp
  src/sg/GlslSourceStitcher.h
  src/sg/GlslSourceStitcher.cpp
  src/sg/MdlGlslCodeGen.h
  src/sg/MdlGlslCodeGen.cpp
  src/sg/MdlLogger.h
  src/sg/MdlLogger.cpp
  src/sg/MdlMaterialCompiler.h
  src/sg/MdlMaterialCompiler.cpp
  src/sg/MdlNeurayLoader.h
  src/sg/MdlNeurayLoader.cpp
  src/sg/MdlRuntime.h
  src/sg/MdlRuntime.cpp
  src/sg/MtlxDocumentPatcher.h
  src/sg/MtlxDocumentPatcher.cpp
  src/sg/MtlxMdlCodeGen.h
  src/sg/MtlxMdlCodeGen.cpp
)

target_include_directories(
  gi
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    src
    shaders
    ${MDL_INCLUDE_DIR}
)

target_compile_definitions(
  gi
  PRIVATE
    GATLING_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GATLING_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GATLING_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    GATLING_SHADER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
)

target_link_libraries(
  gi
  PRIVATE
    gb
    cgpu
    ggpu
    imgio
    glm
    MaterialXCore
    MaterialXFormat
    MaterialXGenMdl
    glslang
    glslang-default-resource-limits
    SPIRV
    efsw-static
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gi PRIVATE OpenMP::OpenMP_CXX)
endif()

# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(ENABLE_IPO)
  set_target_properties(gi PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

install(
  FILES ${MDL_SHARED_LIB}
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mdl/"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources/mdl"
  COMPONENT hdGatling
)
