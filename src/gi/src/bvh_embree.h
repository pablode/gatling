//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

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
