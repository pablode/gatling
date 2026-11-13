set(MC_SRCS
  gtl/mc/Backend.h
  gtl/mc/Frontend.h
  gtl/mc/Material.h
  gtl/mc/MaterialParameters.h
  gtl/mc/MtlxDocOps.h
  gtl/mc/MtlxDocPatch.h
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
  impl/MtlxDocOps.cpp
  impl/MtlxDocPatch.cpp
  impl/MtlxMdlCodeGen.cpp
  impl/MtlxMdlCodeGen.h
  impl/Runtime.cpp
)

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
      gtl/mc
      impl
      ${MDL_INCLUDE_DIR}
  )

  target_link_libraries(
    ${TARGET}
    PRIVATE
      gb
      MaterialXCore
      MaterialXFormat
      MaterialXGenMdl
  )

  if(TARGET OpenColorIO::OpenColorIO)
    target_link_libraries(${TARGET} PUBLIC OpenColorIO::OpenColorIO)
    target_compile_definitions(${TARGET} PUBLIC OCIO)
  endif()
endfunction()

add_library(mc STATIC ${MC_SRCS})
set_target_properties(
  mc
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
configure_target(mc)

add_executable(mc_test ${MC_SRCS} impl/main.cpp)
target_compile_definitions(mc_test PRIVATE MC_TEST_EXECUTABLE=1)
target_link_libraries(mc_test PRIVATE gt doctest)
configure_target(mc_test)
