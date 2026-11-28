set(HDGATLING_SRCS
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

if(TARGET usd_ms)
  set(USD_LIBS_PUBLIC usd_ms) # before v25.11
elseif(TARGET usd_m)
  set(USD_LIBS_PUBLIC usd_m)  # equal and after v25.11
else()
  set(USD_LIBS_PUBLIC hd)
  set(USD_LIBS_PRIVATE usdImaging usdMtlx hdMtlx glf hgi)
endif()

option(HDGATLING_ENABLE_GPU_INTEROP "Render directly into Hgi textures instead of host memory." ON)

function(configure_target TARGET)
  target_include_directories(
    ${TARGET}
    PUBLIC
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
      src
  )

  target_compile_definitions(
    ${TARGET}
    PUBLIC
      # Required for PLUG_THIS_PLUGIN macro
      MFB_PACKAGE_NAME=hdGatling
    PRIVATE
      HDGATLING_LICENSE_FILE_NAME="LICENSE"
      # Workaround for https://github.com/PixarAnimationStudios/USD/issues/1471#issuecomment-799813477
      "$<$<OR:$<CONFIG:>,$<CONFIG:Debug>>:TBB_USE_DEBUG>"
  )

  if (APPLE)
    target_compile_definitions(
      ${TARGET}
      PRIVATE
        # TODO: this feature causes artifacts on Metal which need to be investigated
        HDGATLING_DISABLE_VOLUME_SAMPLING=1
    )
  endif()

  if (HDGATLING_ENABLE_GPU_INTEROP)
    if (APPLE)
      target_compile_definitions(${TARGET} PRIVATE HDGATLING_HGI_INTEROP_METAL=1)
    elseif (TARGET hgiVulkan)
      target_compile_definitions(${TARGET} PRIVATE HDGATLING_HGI_INTEROP_VULKAN=1)
      target_link_libraries(${TARGET} PRIVATE hgiVulkan)
    endif()
  endif()

  # Workaround for https://github.com/PixarAnimationStudios/USD/issues/1279
  if(MSVC_VERSION GREATER_EQUAL 1920)
    target_compile_options(${TARGET} PRIVATE "/Zc:inline-")
  endif()

  target_link_libraries(
    ${TARGET}
    PUBLIC
      ${USD_LIBS_PUBLIC}
    PRIVATE
      gi gb
      ${USD_LIBS_PRIVATE}
  )
endfunction()

add_library(hdGatling SHARED ${HDGATLING_SRCS})

set_target_properties(
  hdGatling
  PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH ON
    # The other libs in the plugin dir have no "lib" prefix, so let's match this
    PREFIX ""
    # Remove _d postfix because of plugInfo.json mismatch on macOS
    DEBUG_POSTFIX ""
)

configure_target(hdGatling)

# Allow custom prefix for houdini directory structure
if(NOT DEFINED PLUGINFO_LIBPATH_PREFIX)
  set(PLUGINFO_LIBPATH_PREFIX "..")
endif()

set(PLUGINFO_PATH "${CMAKE_CURRENT_BINARY_DIR}/plugInfo.json")
set(PLUG_INFO_LIBRARY_PATH "${PLUGINFO_LIBPATH_PREFIX}/hdGatling${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(PXR_VERSION GREATER_EQUAL 2508)
  set(USD_NODE_LIB "Sdr")
else()
  set(USD_NODE_LIB "Ndr")
endif()
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

add_executable(
  hdGatling_test
  main.cpp
  tokens.h
  tokens.cpp
  previewSurfaceNetworkPatcher.cpp
  previewSurfaceNetworkPatcher.h
)
configure_target(hdGatling_test)
target_link_libraries(hdGatling_test PRIVATE gt hdsi hio usd usdGeom usdImaging usdRender)

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
    HDGATLING_TEST_EXECUTABLE=1
)
