#ifndef GP_BVH_H
#define GP_BVH_H

#include <stdbool.h>
#include <gml.h>

#include "gp.h"
#include "math.h"

typedef enum GpBvhBinningMode {
  GP_BVH_BINNING_MODE_ADAPTIVE,
  GP_BVH_BINNING_MODE_FIXED,
  GP_BVH_BINNING_MODE_OFF
} GpBvhBinningMode;

typedef struct gp_bvh_node {
  gml_aabb aabb;
  /* If this node is a leaf, the face offset. Otherwise, the offset
   * to the left child node. */
  uint32_t field1;
  /* If the first bit of this field is set, this node is a leaf.
   * The remaining 31 bits encode the number of faces if the node
   * is a leaf or the offset to the right child node, if it's not. */
  uint32_t field2;
} gp_bvh_node;

typedef struct gp_bvh {
  gml_aabb     aabb;
  uint32_t     node_count;
  gp_bvh_node* nodes;
  uint32_t     face_count;
  gp_face*     faces;
} gp_bvh;

typedef struct gp_bvh_build_params {
  uint32_t         face_batch_size;
  uint32_t         face_count;
  float            face_intersection_cost;
  gp_face*         faces;
  uint32_t         leaf_max_face_count;
  GpBvhBinningMode object_binning_mode;
  uint32_t         object_binning_threshold;
  uint32_t         object_bin_count;
  uint32_t         spatial_bin_count;
  float            spatial_reserve_factor;
  float            spatial_split_alpha;
  uint32_t         vertex_count;
  gp_vertex*       vertices;
} gp_bvh_build_params;

void gp_bvh_build(
  const gp_bvh_build_params* params,
  gp_bvh* bvh
);

void gp_free_bvh(gp_bvh* bvh);

#endif
