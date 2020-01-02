#ifndef GP_BVH_H
#define GP_BVH_H

#include <assert.h>

#include "gp.h"
#include "math.h"

typedef struct gp_bvh_node {
  gp_aabb  left_aabb;          /* 24 bytes */
  uint32_t left_child_index;   /*  4 bytes */
  uint32_t left_child_count;   /*  4 bytes */
  gp_aabb  right_aabb;         /* 24 bytes */
  uint32_t right_child_index;  /*  4 bytes */
  uint32_t right_child_count;  /*  4 bytes */
} gp_bvh_node;

static_assert((sizeof(gp_bvh_node) % 64) == 0, "BVH node should be cache-aligned.");

typedef struct gp_bvh {
  gp_aabb      aabb;
  uint32_t     node_count;
  gp_bvh_node* nodes;
  uint32_t     face_count;
  gp_face*     faces;
  uint32_t     vertex_count;
  gp_vertex*   vertices;
} gp_bvh;

typedef struct gp_bvh_build_params {
  uint32_t   face_count;
  gp_face*   faces;
  uint32_t   min_leaf_size;
  uint32_t   min_mem_fetch_bytes;
  uint32_t   max_leaf_size;
  uint32_t   node_batch_size;
  float      node_traversal_cost;
  uint32_t   sah_bin_count;
  uint32_t   tri_batch_size;
  float      tri_intersection_cost;
  uint32_t   vertex_count;
  gp_vertex* vertices;
} gp_bvh_build_params;

void gp_bvh_build(
  const gp_bvh_build_params* params,
  gp_bvh* bvh
);

void gp_free_bvh(gp_bvh* bvh);

#endif
