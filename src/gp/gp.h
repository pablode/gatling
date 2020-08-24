#ifndef GP_H
#define GP_H

#include <stdint.h>

#if defined(NDEBUG)
  #if defined(__GNUC__)
    #define GP_INLINE inline __attribute__((__always_inline__))
  #elif defined(_MSC_VER)
    #define GP_INLINE __forceinline
  #else
    #define GP_INLINE inline
  #endif
#else
  #define GP_INLINE
#endif

typedef struct gp_vertex {
  float pos[3];
  float norm[3];
  float uv[2];
} gp_vertex;

typedef struct gp_face {
  uint32_t v_i[3];
  uint32_t mat_index;
} gp_face;

typedef struct gp_material {
  float r;
  float g;
  float b;
  float a;
  float emission_r;
  float emission_g;
  float emission_b;
  float padding;
} gp_material;

#endif
