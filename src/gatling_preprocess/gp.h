#pragma once

typedef enum GpResult {
  GP_OK = 0,
  GP_UNABLE_TO_OPEN_FILE = -1,
  GP_UNABLE_TO_CLOSE_FILE = -2,
  GP_UNABLE_TO_IMPORT_SCENE = -3
} GpResult;

typedef struct gp_vertex {
  float pos_x;
  float pos_y;
  float pos_z;
  float norm_x;
  float norm_y;
  float norm_z;
  float t_u;
  float t_v;
} gp_vertex;

typedef struct gp_triangle {
  uint32_t v0;
  uint32_t v1;
  uint32_t v2;
  uint32_t mat_index;
} gp_triangle;
