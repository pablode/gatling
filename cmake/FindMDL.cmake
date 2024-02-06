#
# Find MDL.
#
# Once done, this will define following vars:
# MDL_FOUND
# MDL_INCLUDE_DIR
# MDL_SHARED_LIB
# MDL_LIB_DIR
#

find_path(MDL_INCLUDE_DIR
  NAMES mi/mdl_sdk.h
  PATH_SUFFIXES include
  HINTS ${MDL_ROOT}
)

if(WIN32)
  set(MDL_OS_LIB_DIR "nt-x86-64")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")
elseif(APPLE)
  set(MDL_OS_LIB_DIR "macosx-uni")
else()
  set(MDL_OS_LIB_DIR "linux-x86-64")
endif()

find_library(MDL_SHARED_LIB
  NAMES mdl_sdk libmdl_sdk
  HINTS ${MDL_ROOT}
  PATH_SUFFIXES bin lib ${MDL_OS_LIB_DIR}/lib
)

get_filename_component(MDL_LIB_DIR ${MDL_SHARED_LIB} DIRECTORY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MDL
  DEFAULT_MSG
  MDL_INCLUDE_DIR
  MDL_SHARED_LIB
  MDL_LIB_DIR
)

mark_as_advanced(MDL_INCLUDE_DIR MDL_SHARED_LIB MDL_LIB_DIR)
