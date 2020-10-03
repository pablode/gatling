# On some systems, we have to explicitly link against the
# C runtime math library (-lm). The only way to find out
# whether it's actually needed or not is by testing it.

include(CheckCSourceCompiles)

set(C_MATH_LIBRARY_TEST_SOURCE
  "#include <math.h>
   int main() {
    sinf(0.0f);
    powf(1.0f, 2.0f);
   }")

set(CMAKE_REQUIRED_FLAGS "")
set(CMAKE_REQUIRED_LIBRARIES "")
set(CMAKE_REQUIRED_QUIET TRUE)

check_c_source_compiles(
  "${C_MATH_LIBRARY_TEST_SOURCE}"
  C_MATH_LIBRARY_AUTO_LINK_OK
)

if(C_MATH_LIBRARY_AUTO_LINK_OK)
  set(C_MATH_LIBRARY "")
else()
  find_library(C_MATH_LIBRARY "m")

  set(CMAKE_REQUIRED_LIBRARIES "${C_MATH_LIBRARY}")

  check_c_source_compiles(
    "${C_MATH_LIBRARY_TEST_SOURCE}"
    C_MATH_LIBRARY_MANUAL_LINK_OK
  )

  if(NOT C_MATH_LIBARY_MANUAL_LINK_OK)
    message(FATAL_ERROR "Unable to link against C math library.")
  endif()
endif()
