add_library(
  hdGatling SHARED
  Instancer.cpp
  Instancer.h
  Light.cpp
  Light.h
  Material.cpp
  Material.h
  MaterialNetworkCompiler.cpp
  MaterialNetworkCompiler.h
  MdlDiscoveryPlugin.cpp
  MdlDiscoveryPlugin.h
  MdlParserPlugin.cpp
  MdlParserPlugin.h
  Mesh.cpp
  Mesh.h
  PreviewSurfaceNetworkPatcher.cpp
  PreviewSurfaceNetworkPatcher.h
  RenderBuffer.cpp
  RenderBuffer.h
  RenderDelegate.cpp
  RenderDelegate.h
  RendererPlugin.cpp
  RendererPlugin.h
  RenderParam.cpp
  RenderParam.h
  RenderPass.cpp
  RenderPass.h
  Tokens.cpp
  Tokens.h
)

set_target_properties(
  hdGatling
  PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    # The other libs in the plugin dir have no "lib" prefix, so let's match this
    PREFIX ""
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
    LICENSE_FILE_NAME="LICENSE"
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
    gi
    ${USD_LIBS_PRIVATE}
)

set(PLUGINFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json")
set(PLUG_INFO_LIBRARY_PATH "../hdGatling${CMAKE_SHARED_LIBRARY_SUFFIX}")
configure_file(plugInfo.json.in "${PLUGINFO_PATH}" @ONLY)

install(
  FILES
    "${PLUGINFO_PATH}"
    "${PROJECT_SOURCE_DIR}/LICENSE"
  DESTINATION
    "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
  COMPONENT
    hdGatling
)

install(
  TARGETS
    hdGatling
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
    COMPONENT hdGatling
  RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
    COMPONENT hdGatling
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_PREFIX}"
    COMPONENT hdGatling
)
