add_library(
  mdl-image-plugin MODULE
  plugin.cpp
  plugin.h
)

set_target_properties(
  mdl-image-plugin
  PROPERTIES
    # The other libs in the plugin dir have no "lib" prefix, so let's match this
    PREFIX ""
)

target_compile_definitions(
  mdl-image-plugin
  PRIVATE
    # declspec(dllexport) macro
    MI_DLL_BUILD
)

target_include_directories(
  mdl-image-plugin
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${MDL_INCLUDE_DIR}
)

install(
  TARGETS
    mdl-image-plugin
  LIBRARY
    DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
    COMPONENT hdGatling
  RUNTIME
    DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
    COMPONENT hdGatling
  ARCHIVE
    DESTINATION "${CMAKE_INSTALL_PREFIX}/hdGatling/resources"
    COMPONENT hdGatling
)
