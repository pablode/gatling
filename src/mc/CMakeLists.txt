add_library(
  mc STATIC
  gtl/mc/Backend.h
  gtl/mc/Frontend.h
  gtl/mc/Material.h
  gtl/mc/MtlxDocOps.h
  gtl/mc/MtlxDocPatch.h
  gtl/mc/Runtime.h
  impl/Backend.cpp
  impl/Frontend.cpp
  impl/Material.cpp
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
  impl/MtlxDocOps.cpp
  impl/MtlxDocPatch.cpp
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

if(DEFINED PXR_VERSION)
  target_compile_definitions(
    mc
    PRIVATE
      PXR_VERSION=${PXR_VERSION}
  )
endif()

# Required since library is linked into hdGatling DSO
set_target_properties(mc PROPERTIES POSITION_INDEPENDENT_CODE ON)
