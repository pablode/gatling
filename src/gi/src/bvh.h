#ifndef GI_BVH_H
#define GI_BVH_H

#include <gml.h>

struct gi_face;
struct gi_vertex;

enum GiBvhBinningMode
{
  GI_BVH_BINNING_MODE_ADAPTIVE,
  GI_BVH_BINNING_MODE_FIXED,
  GI_BVH_BINNING_MODE_OFF
};

struct gi_bvh_node
{
  gml_aabb aabb;
  /* If this node is a leaf, the face offset. Otherwise, the offset
   * to the left child node. */
  uint32_t field1;
  /* If the first bit of this field is set, this node is a leaf.
   * The remaining 31 bits encode the number of faces if the node
   * is a leaf or the offset to the right child node, if it's not. */
  uint32_t field2;
};

struct gi_bvh
{
  gml_aabb     aabb;
  uint32_t     node_count;
  gi_bvh_node* nodes;
  uint32_t     face_count;
  gi_face*     faces;
};

struct gi_bvh_params {
  uint32_t         face_batch_size;
  uint32_t         face_count;
  float            face_intersection_cost;
  gi_face*         faces;
  uint32_t         leaf_max_face_count;
  GiBvhBinningMode object_binning_mode;
  uint32_t         object_binning_threshold;
  uint32_t         object_bin_count;
  uint32_t         spatial_bin_count;
  float            spatial_reserve_factor;
  float            spatial_split_alpha;
  uint32_t         vertex_count;
  gi_vertex*       vertices;
};

void gi_bvh_build(const gi_bvh_params* params,
                  gi_bvh* bvh);

void gi_free_bvh(gi_bvh* bvh);

#endif
