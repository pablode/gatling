add_library(
  gb STATIC
  gtl/gb/Class.h
  gtl/gb/Enum.h
  gtl/gb/Fmt.h
  gtl/gb/HandleStore.h
  gtl/gb/LinearDataStore.h
  gtl/gb/Log.h
  gtl/gb/ParamTypes.h
  gtl/gb/SmallVector.h
  impl/HandleStore.cpp
  impl/LinearDataStore.cpp
  impl/Log.cpp
  impl/SmallVector.cpp
)

target_include_directories(
  gb
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    gtl/gb
    impl
)

target_link_libraries(
  gb
  PUBLIC
    quill
)

# Required since library is linked into hdGatling DSO
set_target_properties(gb PROPERTIES POSITION_INDEPENDENT_CODE ON)
