#pragma once

#include <stddef.h>
#include <vector>

#include "bvh.h"

namespace gi
{
  namespace bvh
  {
    template<size_t WIDTH>
    struct BvhNode
    {
      gml_aabb aabbs[WIDTH];
      uint32_t offsets[WIDTH];
      uint32_t counts[WIDTH];
      uint32_t child_index;
      uint32_t face_index;
    };

    template<size_t WIDTH>
    struct Bvh
    {
      gml_aabb                    aabb;
      std::vector<BvhNode<WIDTH>> nodes;
      std::vector<gi_face>        faces;
    };

    struct CollapseParams
    {
      float    face_intersection_cost;
      float    node_traversal_cost;
      uint32_t max_leaf_size;
    };

    template<size_t N>
    bool collapse_bvh2(const Bvh2& bvh2,
                       const CollapseParams& params,
                       Bvh<N>& bvh);
  }
}

#include "bvh_collapse.inl"
