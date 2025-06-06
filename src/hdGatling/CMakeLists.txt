add_library(
  hdGatling SHARED
  instancer.cpp
  instancer.h
  light.cpp
  light.h
  material.cpp
  material.h
  materialNetworkCompiler.cpp
  materialNetworkCompiler.h
  mdlDiscoveryPlugin.cpp
  mdlDiscoveryPlugin.h
  mdlParserPlugin.cpp
  mdlParserPlugin.h
  mesh.cpp
  mesh.h
  previewSurfaceNetworkPatcher.cpp
  previewSurfaceNetworkPatcher.h
  renderBuffer.cpp
  renderBuffer.h
  renderDelegate.cpp
  renderDelegate.h
  rendererPlugin.cpp
  rendererPlugin.h
  renderParam.cpp
  renderParam.h
  renderPass.cpp
  renderPass.h
  tokens.cpp
  tokens.h
  utils.cpp
  utils.h
)

set_target_properties(
  hdGatling
  PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    # The other libs in the plugin dir have no "lib" prefix, so let's match this
    PREFIX ""
    # Remove _d postfix because of plugInfo.json mismatch on macOS
    DEBUG_POSTFIX ""
)

target_include_directories(
  hdGatling
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    src
)

target_compile_definitions(
  hdGatling
  PUBLIC
    # Required for PLUG_THIS_PLUGIN macro
    MFB_PACKAGE_NAME=hdGatling
  PRIVATE
    GI_LICENSE_FILE_NAME="LICENSE"
    # Workaround for https://github.com/PixarAnimationStudios/USD/issues/1471#issuecomment-799813477
    "$<$<OR:$<CONFIG:>,$<CONFIG:Debug>>:TBB_USE_DEBUG>"
)

# Workaround for https://github.com/PixarAnimationStudios/USD/issues/1279
if(MSVC_VERSION GREATER_EQUAL 1920)
  target_compile_options(hdGatling PRIVATE "/Zc:inline-")
endif()

if(TARGET usd_ms)
  set(USD_LIBS_PUBLIC usd_ms)
else()
  set(USD_LIBS_PUBLIC hd)
  set(USD_LIBS_PRIVATE usdImaging usdMtlx hdMtlx glf)
endif()

target_link_libraries(
  hdGatling
  PUBLIC
    ${USD_LIBS_PUBLIC}
  PRIVATE
    gi gb
    ${USD_LIBS_PRIVATE}
)

# Allow custom prefix for houdini directory structure
if(NOT DEFINED PLUGINFO_LIBPATH_PREFIX)
  set(PLUGINFO_LIBPATH_PREFIX "..")
endif()

set(PLUGINFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json")
set(PLUG_INFO_LIBRARY_PATH "${PLUGINFO_LIBPATH_PREFIX}/hdGatling${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(plugInfo.json.in "${PLUGINFO_PATH}" @ONLY)

install(
  FILES
    "${PLUGINFO_PATH}"
    "${PROJECT_SOURCE_DIR}/LICENSE"
  DESTINATION
    "./hdGatling/resources"
  COMPONENT
    hdGatling
)

install(
  TARGETS
    hdGatling
  LIBRARY
    DESTINATION .
    COMPONENT hdGatling
  RUNTIME
    DESTINATION .
    COMPONENT hdGatling
  ARCHIVE
    DESTINATION .
    COMPONENT hdGatling
)

add_executable(hdGatling_test main.cpp tokens.h tokens.cpp)
target_link_libraries(hdGatling_test gt gb hd hio usd usdGeom usdImaging usdRender)

add_dependencies(hdGatling_test hdGatling)

# Test executable operates on hdGatling installed to a local directory
set(HDGATLING_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/INSTALL")
add_custom_command(
  TARGET hdGatling_test
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} --install "${CMAKE_BINARY_DIR}" --prefix "${HDGATLING_INSTALL_DIR}" --component hdGatling
)

target_compile_definitions(
  hdGatling_test
  PRIVATE
    HDGATLING_INSTALL_DIR="${HDGATLING_INSTALL_DIR}"
    HDGATLING_TESTENV_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testenv"
    HDGATLING_TEST_OUTPUT_DIR="${GTL_TEST_OUTPUT_DIR}/hdGatling"
)
