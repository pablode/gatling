add_library(
  hdGatling SHARED
  include/RendererPlugin.h
  src/RendererPlugin.cpp
  src/RenderDelegate.h
  src/RenderDelegate.cpp
  src/Camera.h
  src/Camera.cpp
  src/Material.h
  src/Material.cpp
  src/Mesh.h
  src/Mesh.cpp
  src/RenderPass.h
  src/RenderPass.cpp
  src/Instancer.h
  src/Instancer.cpp
  src/RenderBuffer.h
  src/RenderBuffer.cpp
  src/Settings.h
  src/Settings.cpp
)

set_target_properties(
  hdGatling
  PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
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
    gi usdImaging
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
    hdGatling gi
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
