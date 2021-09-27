#
# Find DXC.
#
# Once done, this will define the DXC target and following vars:
# DXC_FOUND
# DXC_INCLUDE_DIRS
# DXC_LIBRARIES
#

find_path(DXC_INCLUDE_DIRS
  NAMES dxcapi.h
  PATH_SUFFIXES inc
  HINTS ${DXC_ROOT})

find_library(DXC_LIBRARIES
  NAMES dxcompiler
  PATH_SUFFIXES lib/x64
  HINTS ${DXC_ROOT})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DXC
  DEFAULT_MSG
  DXC_INCLUDE_DIRS
  DXC_LIBRARIES)

mark_as_advanced(DXC_INCLUDE_DIRS DXC_LIBRARIES)

if(DXC_FOUND AND NOT TARGET DXC)
  add_library(DXC UNKNOWN IMPORTED)
  set_target_properties(DXC PROPERTIES
    IMPORTED_LOCATION "${DXC_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${DXC_INCLUDE_DIRS}")
endif()
