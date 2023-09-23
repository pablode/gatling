add_library(
  gi STATIC
  include/gi.h
  src/gi.cpp
  src/assetReader.h
  src/assetReader.cpp
  src/mmap.h
  src/mmap.cpp
  src/textureManager.h
  src/textureManager.cpp
  src/turbo.h
  src/GlslShaderCompiler.h
  src/GlslShaderCompiler.cpp
  src/GlslShaderGen.h
  src/GlslShaderGen.cpp
  src/GlslStitcher.h
  src/GlslStitcher.cpp
)

target_include_directories(
  gi
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    src
    shaders
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
    mc
    cgpu
    ggpu
    imgio
    glm
    MaterialXCore
    efsw-static
    glslang
    glslang-default-resource-limits
    SPIRV
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(gi PRIVATE OpenMP::OpenMP_CXX)
endif()

# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
