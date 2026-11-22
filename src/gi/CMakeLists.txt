set(GI_SRCS
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
  impl/SceneDataPool.h
  impl/SceneDataPool.cpp
  impl/TextureManager.h
  impl/TextureManager.cpp
  impl/Turbo.h
)

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
      gtl/gi
      impl
      shaders
  )

  target_compile_definitions(
    ${TARGET}
    PRIVATE
      GI_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
      GI_VERSION_MINOR=${PROJECT_VERSION_MINOR}
      GI_VERSION_PATCH=${PROJECT_VERSION_PATCH}
      GI_SHADER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  )

  target_link_libraries(
    ${TARGET}
    PUBLIC
      gb
    PRIVATE
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
      xxHash
  )
endfunction()

add_library(gi STATIC ${GI_SRCS})
set_target_properties(
  gi
  PROPERTIES
    # Required since library is linked into hdGatling DSO
    POSITION_INDEPENDENT_CODE ON
)
configure_target(gi)

add_executable(gi_test ${GI_SRCS} impl/main.cpp)
target_compile_definitions(gi_test PRIVATE GI_TEST_EXECUTABLE=1)
target_link_libraries(gi_test PRIVATE gt doctest)
configure_target(gi_test)

install(
  FILES "${MDL_SHARED_LIB}"
  DESTINATION "./hdGatling/resources"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  DESTINATION "./hdGatling/resources"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mdl/"
  DESTINATION "./hdGatling/resources/mdl"
  COMPONENT hdGatling
)

install(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/mtlx/"
  DESTINATION "./hdGatling/resources/mtlx"
  COMPONENT hdGatling
)
