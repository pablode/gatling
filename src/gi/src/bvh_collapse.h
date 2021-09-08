#ifndef GP_BVH_COLLAPSE_H
#define GP_BVH_COLLAPSE_H

#include <stddef.h>

#include "bvh.h"

struct gp_bvhc_node
{
  gml_aabb aabbs[8];
  uint32_t offsets[8];
  uint32_t counts[8];
  uint32_t child_index;
  uint32_t face_index;
};

struct gp_bvhc
{
  gml_aabb             aabb;
  uint32_t             node_count;
  struct gp_bvhc_node* nodes;
  uint32_t             face_count;
  struct gi_face*      faces;
};

struct gp_bvh_collapse_params
{
  const struct gp_bvh* bvh;
  float                face_intersection_cost;
  uint32_t             max_leaf_size;
  float                node_traversal_cost;
};

void gp_bvh_collapse(const struct gp_bvh_collapse_params* params,
                     struct gp_bvhc* bvhc);

void gp_free_bvhc(struct gp_bvhc* bvhcc);

#endif
