set(GI_SRCS
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

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
      src
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
  )

  if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_library(gi STATIC ${GI_SRCS})
target_link_libraries(gi PRIVATE efsw-static)
# Required since library is linked into hdGatling DSO
set_target_properties(gi PROPERTIES POSITION_INDEPENDENT_CODE ON)
configure_target(gi)

add_executable(gi_test ${GI_SRCS} src/main.cpp)
target_link_libraries(gi_test PRIVATE gt doctest)
configure_target(gi_test)

target_compile_definitions(
  gi_test
  PRIVATE
    GI_TEST_EXECUTABLE
    GI_MDL_LIB_DIR="${MDL_LIB_DIR}"
    GI_MTLX_STDLIB_DIR="${MATERIALX_STDLIB_DIR}"
    GI_REF_IMAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testenv"
)

# Expose tests using ctest for IDE integration
include(CTest)
include("${DOCTEST_SCRIPT_PATH}")
doctest_discover_tests(gi_test WORKING_DIRECTORY "${OUTPUT_DIR}")

install(
  FILES "${MDL_SHARED_LIB}"
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
