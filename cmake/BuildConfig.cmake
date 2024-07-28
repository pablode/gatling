# Use RelWithDebInfo as base config
get_directory_property(_vars VARIABLES)
foreach(_var IN LISTS _vars)
    if (_var MATCHES "_RELWITHDEBINFO$")
        string(REPLACE "_RELWITHDEBINFO" "_DEVELOP" _var_new "${_var}")
        set(${_var_new} "${${_var}}")
        mark_as_advanced(${_var_new})
    endif()
endforeach()

# But with assertions
if(MSVC)
  string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_DEVELOP "${CMAKE_CXX_FLAGS_DEVELOP}")
  string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_DEVELOP "${CMAKE_C_FLAGS_DEVELOP}")
else()
  string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_DEVELOP "${CMAKE_CXX_FLAGS_DEVELOP}")
  string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_DEVELOP "${CMAKE_C_FLAGS_DEVELOP}")
endif()

# Update the documentation string for GUIs
set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel Develop." FORCE)

# Append list of configs for multi-output generators like VS
list(APPEND CMAKE_CONFIGURATION_TYPES "Develop")


# TODO: define GTL_COMPILE_OPTIONS that is then manually set for each target
# TODO: define GTL_LINK_OPTIONS -> target_link_options (CMake 3.13). e.g. warnings as errors?
if(MSVC)
  set(GTL_COMPILE_OPTIONS "")#"/Wall")
  # Disable "not enough arguments for function-like macro invocation" warning caused
  # by the transitive inclusion of boost headers and non-conformant MSVC preprocessor
  # behaviour. See USD's msvcdefaults.cmake file for a detailed comment.
  set(GTL_COMPILE_OPTIONS "${GTL_COMPILE_OPTIONS}" "/wd4003")
else()
  set(GTL_COMPILE_OPTIONS "-Wall" "-Wextra" "-Wformat-security")
endif()

# TODO: for headers, either use a #pragma OR in case of USD/pxr, add to the target additionally


