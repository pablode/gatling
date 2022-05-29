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
