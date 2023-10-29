add_library(
  mc STATIC
  include/Backend.h
  include/Frontend.h
  include/Material.h
  include/Runtime.h
  src/Backend.cpp
  src/Frontend.cpp
  src/Material.cpp
  src/MdlLogger.cpp
  src/MdlLogger.h
  src/MdlMaterial.cpp
  src/MdlMaterial.h
  src/MdlMaterialCompiler.cpp
  src/MdlMaterialCompiler.h
  src/MdlNeurayLoader.cpp
  src/MdlNeurayLoader.h
  src/MdlRuntime.cpp
  src/MdlRuntime.h
  src/MtlxDocumentPatcher.cpp
  src/MtlxDocumentPatcher.h
  src/MtlxMdlCodeGen.cpp
  src/MtlxMdlCodeGen.h
  src/Runtime.cpp
)

target_include_directories(
  mc
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    src
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
