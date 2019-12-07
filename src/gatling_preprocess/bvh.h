#pragma once

#include <stdint.h>

#include "gp.h"

typedef struct gp_bvh_node {
  float min_x;
  float min_y;
  float min_z;
  uint32_t child_nodes_index_offset;
  float max_x;
  float max_y;
  float max_z;
  uint32_t leaf_node_count;
} gp_bvh_node;

typedef struct gp_bvh {
  gp_bvh_node* nodes;
  uint32_t node_count;
  gp_vertex vertices;
  uint32_t vertices_count;
} gp_bvh;

typedef struct gp_bvh_build_input
{
  const gp_vertex* vertices;
  uint32_t vertices_count;
  const gp_triangle* triangles;
  uint32_t triangles_count;
  // The spatial split factor lies within [0, 1] and denotes the
  // overlap area to root area ratio which is tolerated without
  // attempting a spatial split (0.0 = full sbvh, 1.0 = regular BVH).
  // It is recommended for this value to be close to zero (e.g. 10^-5).
  // More information can be found in the paper "Spatial Splits in
  // Bounding Volume Hierarchies" by Stich, Friedrich and Dietrich ('09).
  float spatial_split_alpha;
} gp_bvh_build_input;

GpResult gp_bvh_build(
  const gp_bvh_build_input* input,
  gp_bvh* bvh
);
