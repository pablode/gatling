#ifndef GP_BVH_COLLAPSE_H
#define GP_BVH_COLLAPSE_H

#include <stddef.h>

#include "bvh.h"

typedef struct gp_bvhc_node {
  gp_aabb  aabbs[8];
  uint32_t offsets[8];
  uint32_t counts[8];
  uint32_t child_index;
  uint32_t face_index;
} gp_bvhc_node;

typedef struct gp_bvhc {
  gp_aabb       aabb;
  uint32_t      node_count;
  gp_bvhc_node* nodes;
  uint32_t      face_count;
  gp_face*      faces;
} gp_bvhc;

typedef struct gp_bvh_collapse_params {
  const gp_bvh* bvh;
  float         face_intersection_cost;
  uint32_t      max_leaf_size;
  float         node_traversal_cost;
} gp_bvh_collapse_params;

void gp_bvh_collapse(const gp_bvh_collapse_params* params, gp_bvhc* bvhc);

void gp_free_bvhc(gp_bvhc* bvhcc);

#endif
