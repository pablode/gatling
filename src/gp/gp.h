#ifndef GP_H
#define GP_H

#include <stdint.h>

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
  float albedo_r;
  float albedo_g;
  float albedo_b;
  float padding1;
  float emission_r;
  float emission_g;
  float emission_b;
  float padding2;
} gp_material;

#endif
