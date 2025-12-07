add_library(
  mc STATIC
  gtl/mc/Backend.h
  gtl/mc/Frontend.h
  gtl/mc/Material.h
  gtl/mc/MaterialParameters.h
  gtl/mc/Runtime.h
  impl/Backend.cpp
  impl/Frontend.cpp
  impl/Material.cpp
  impl/MdlEntityResolver.cpp
  impl/MdlEntityResolver.h
  impl/MdlLogger.cpp
  impl/MdlLogger.h
  impl/MdlMaterial.cpp
  impl/MdlMaterial.h
  impl/MdlMaterialCompiler.cpp
  impl/MdlMaterialCompiler.h
  impl/MdlNeurayLoader.cpp
  impl/MdlNeurayLoader.h
  impl/MdlRuntime.cpp
  impl/MdlRuntime.h
  impl/MtlxDocumentPatcher.cpp
  impl/MtlxDocumentPatcher.h
  impl/MtlxMdlCodeGen.cpp
  impl/MtlxMdlCodeGen.h
  impl/Runtime.cpp
)

target_include_directories(
  mc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/mc
    impl
    ${MDL_INCLUDE_DIR}
)

target_link_libraries(
  mc
  PRIVATE
    gb
    MaterialXCore
    MaterialXFormat
    MaterialXGenMdl
)

math(EXPR MATERIALX_VERSION "${MaterialX_VERSION_MAJOR} * 100 * 100 +
                             ${MaterialX_VERSION_MINOR} * 100 +
                             ${MaterialX_VERSION_PATCH}")
target_compile_definitions(
  mc
  PRIVATE
    MATERIALX_VERSION=${MATERIALX_VERSION}
)

if(DEFINED PXR_VERSION)
  target_compile_definitions(
    mc
    PRIVATE
      PXR_VERSION=${PXR_VERSION}
  )
endif()

# Required since library is linked into hdGatling DSO
set_target_properties(mc PROPERTIES POSITION_INDEPENDENT_CODE ON)
