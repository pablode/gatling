add_library(
  gi STATIC
  gtl/gi/Gi.h
  impl/Gi.cpp
  impl/AssetReader.h
  impl/AssetReader.cpp
  impl/GlslShaderCompiler.h
  impl/GlslShaderCompiler.cpp
  impl/GlslShaderGen.h
  impl/GlslShaderGen.cpp
  impl/GlslStitcher.h
  impl/GlslStitcher.cpp
  impl/Mmap.h
  impl/Mmap.cpp
  impl/MeshProcessing.h
  impl/MeshProcessing.cpp
  impl/Oidn.cpp
  impl/Oidn.h
  impl/TextureManager.h
  impl/TextureManager.cpp
  impl/Tza.h
  impl/Tza.cpp
  impl/Turbo.h
)

target_include_directories(
  gi
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/gi
    impl
    shaders
)

target_compile_definitions(
  gi
  PRIVATE
    GI_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    GI_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    GI_VERSION_PATCH=${PROJECT_VERSION_PATCH}
    GI_SHADER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    GI_OIDN_WEIGHTS_FILE="${GTL_OIDN_WEIGHTS_FILE}"
)

target_link_libraries(
  gi
  PRIVATE
    gb
    mc
    cgpu
    ggpu
    imgio
    glm
    MaterialXCore
    MaterialXFormat
    glslang
    glslang-default-resource-limits
    SPIRV
    meshoptimizer
    blosc2_static
    efsw-static
    OffsetAllocator
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gi PRIVATE OpenMP::OpenMP_CXX)
endif()

# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(
  FILES "${MDL_SHARED_LIB}"
  DESTINATION "${GTL_HDGATLING_RESOURCES_DIR}"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  DESTINATION "${GTL_HDGATLING_RESOURCES_DIR}"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mdl/"
  DESTINATION "${GTL_HDGATLING_RESOURCES_DIR}/mdl"
  COMPONENT hdGatling
)
