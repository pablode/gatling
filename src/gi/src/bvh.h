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

#include <gml.h>
#include <vector>

struct gi_face;
struct gi_vertex;

namespace gi
{
  namespace bvh
  {
    struct BvhNode2
    {
      gml_aabb aabb;
      // If this node is a leaf, the face offset. Otherwise, the offset
      // to the left child node.
      uint32_t field1;
      // If the first bit of this field is set, this node is a leaf.
      // The remaining 31 bits encode the number of faces if the node
      // is a leaf or the offset to the right child node, if it's not.
      uint32_t field2;
    };

    struct Bvh2
    {
      std::vector<BvhNode2> nodes;
      std::vector<gi_face> faces;
    };

    enum class BvhBinningMode
    {
      Adaptive,
      Fixed,
      Off
    };

    struct BvhBuildParams {
      uint32_t       face_batch_size;
      uint32_t       face_count;
      float          face_intersection_cost;
      gi_face*       faces;
      uint32_t       leaf_max_face_count;
      BvhBinningMode object_binning_mode;
      uint32_t       object_binning_threshold;
      uint32_t       object_bin_count;
      uint32_t       spatial_bin_count;
      float          spatial_split_alpha;
      uint32_t       vertex_count;
      gi_vertex*     vertices;
    };

    Bvh2 build_bvh2(const BvhBuildParams& params);
  }
}
