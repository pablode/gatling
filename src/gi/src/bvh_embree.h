#pragma once

#include "bvh.h"

struct gi_vertex;

namespace gi
{
  namespace bvh
  {
    struct EmbreeBuildParams {
      uint32_t   face_batch_size;
      uint32_t   face_count;
      float      face_intersection_cost;
      gi_face*   faces;
      float      node_traversal_cost;
      uint32_t   vertex_count;
      gi_vertex* vertices;
    };

    Bvh2 build_bvh2_embree(const EmbreeBuildParams& params);
  }
}
