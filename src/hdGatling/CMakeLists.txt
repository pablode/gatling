add_library(
  hdGatling MODULE
  Camera.cpp
  Camera.h
  Instancer.cpp
  Instancer.h
  Material.cpp
  Material.h
  MaterialNetworkTranslator.cpp
  MaterialNetworkTranslator.h
  MdlDiscoveryPlugin.cpp
  MdlDiscoveryPlugin.h
  MdlParserPlugin.cpp
  MdlParserPlugin.h
  Mesh.cpp
  Mesh.h
  RenderBuffer.cpp
  RenderBuffer.h
  RenderDelegate.cpp
  RenderDelegate.h
  RendererPlugin.cpp
  RendererPlugin.h
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
    # Workaround for https://github.com/PixarAnimationStudios/USD/issues/1471#issuecomment-799813477
    "$<$<OR:$<CONFIG:>,$<CONFIG:Debug>>:TBB_USE_DEBUG>"
)

# Workaround for https://github.com/PixarAnimationStudios/USD/issues/1279
if (MSVC_VERSION GREATER_EQUAL 1920)
    target_compile_options(hdGatling PRIVATE "/Zc:inline-")
endif()

target_link_libraries(
  hdGatling
  PRIVATE
    usdImaging
    hdMtlx
    gi
  PUBLIC
    hd
)

set(PLUGINFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json")

file(READ plugInfo.json.in PLUGINFO)
file(
  GENERATE
  OUTPUT
    "${PLUGINFO_PATH}"
  CONTENT
    ${PLUGINFO}
)

add_dependencies(hdGatling mdl-image-plugin)

install(
  FILES
    "${PLUGINFO_PATH}"
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
