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

#include "gi.h"

#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <array>

//
// This file implements construction of an N-wide BVH from a binary BVH as described
// by Ylitie, Karras and Laine.
// It works by first calculating SAH costs for representing the contents of each subtree
// as a forest of at most i BVHs. By doing this bottom-up, previous results can be reused.
// For each node and subtree count, we store the minimal cost in an N * (I-1) table, where
// N is the number of nodes and I is the width of the BVH.
// In a second pass, we traverse top-down and trace the decisions leading to the minimal
// costs stored in the table. We inline DISTRIBUTE splits and combine leaf nodes. For each
// INTERNAL split decision, we recurse further down.
//
namespace gi
{
  namespace bvh
  {
    namespace detail
    {
      enum class SplitType
      {
        Invalid = 0,
        Internal,
        Leaf,
        Distribute
      };

      struct Split
      {
        SplitType type;
        int32_t left_count;
        int32_t right_count;
        float cost;
      };

      template<size_t N>
      struct WorkData
      {
        const Bvh2& bvh2;
        Bvh<N>& bvh;
        const CollapseParams& params;
        std::vector<Split> splits;

        WorkData(const Bvh2& bvh2, Bvh<N>& bvh, const CollapseParams& params)
          : bvh2(bvh2), bvh(bvh), params(params)
        {
        }
      };

      template<size_t N>
      uint32_t count_child_faces(const WorkData<N>& wdata, uint32_t node_idx)
      {
        const BvhNode2& node = wdata.bvh2.nodes[node_idx];

        if ((node.field2 & 0x80000000) == 0x80000000)
        {
          return (node.field2 & 0x7FFFFFFF);
        }

        return count_child_faces(wdata, node.field1) + count_child_faces(wdata, node.field2);
      }

      // We need to forward-declare this function.
      template<size_t N>
      Split C(const WorkData<N>& wdata, uint32_t n, uint32_t i);

      template<size_t N>
      Split C_distribute(const WorkData<N>& wdata, uint32_t n, uint32_t j)
      {
        const BvhNode2& node = wdata.bvh2.nodes[n];

        Split split;
        split.type = SplitType::Distribute;
        split.cost = INFINITY;

        for (uint32_t k = 0; k < j; k++)
        {
          Split split_left = C(wdata, node.field1, k);
          Split split_right = C(wdata, node.field2, j - k - 1);
          float cost = split_left.cost + split_right.cost;

          if (cost < split.cost)
          {
            split.cost = cost;
            split.left_count = k;
            split.right_count = j - k - 1;
          }
        }

        return split;
      }

      template<size_t N>
      Split C_internal(const WorkData<N>& wdata, uint32_t n)
      {
        const BvhNode2& node = wdata.bvh2.nodes[n];
        float A_n = gml_aabb_area(&node.aabb);

        Split split = C_distribute(wdata, n, N - 1);
        split.type = SplitType::Internal;
        split.cost += A_n * wdata.params.node_traversal_cost;
        return split;
      }

      template<size_t N>
      Split C_leaf(const WorkData<N>& wdata, uint32_t n)
      {
        uint32_t p_n = count_child_faces(wdata, n);

        Split split;
        split.type = SplitType::Leaf;

        if (p_n > wdata.params.max_leaf_size)
        {
          split.cost = INFINITY;
          return split;
        }

        const BvhNode2& node = wdata.bvh2.nodes[n];
        float A_n = gml_aabb_area(&node.aabb);
        split.cost = A_n * p_n * wdata.params.face_intersection_cost;

        return split;
      }

      template<size_t N>
      Split C(const WorkData<N>& wdata, uint32_t n, uint32_t i)
      {
        if (wdata.splits[n * (N - 1) + i].type != SplitType::Invalid)
        {
          return wdata.splits[n * (N - 1) + i];
        }

        if (i == 0)
        {
          Split c_leaf = C_leaf(wdata, n);
          Split c_internal = C_internal(wdata, n);
          return (c_leaf.cost < c_internal.cost) ? c_leaf : c_internal;
        }
        else
        {
          Split c_dist = C_distribute(wdata, n, i);
          Split c_recur = C(wdata, n, i - 1);
          return (c_dist.cost < c_recur.cost) ? c_dist : c_recur;
        }
      }

      template<size_t N>
      void calc_costs(WorkData<N>& wdata, uint32_t n)
      {
        const BvhNode2 node = wdata.bvh2.nodes[n];

        if ((node.field2 & 0x80000000) == 0x80000000)
        {
          float A_n = gml_aabb_area(&node.aabb);
          uint32_t p_n = (node.field2 & 0x7FFFFFFF);
          float cost = A_n * p_n * wdata.params.face_intersection_cost;

          for (uint32_t i = 0; i < (N - 1); ++i)
          {
            wdata.splits[n * (N - 1) + i].type = SplitType::Leaf;
            wdata.splits[n * (N - 1) + i].cost = cost;
          }
          return;
        }

        calc_costs(wdata, node.field1);
        calc_costs(wdata, node.field2);

        for (uint32_t i = 0; i < (N - 1); ++i)
        {
          wdata.splits[n * (N - 1) + i] = C(wdata, n, i);
        }
      }

      template<size_t N>
      void collect_childs(const WorkData<N>& wdata,
                          uint32_t node_idx,
                          uint32_t child_idx,
                          uint32_t& child_count,
                          std::array<uint32_t, N>& child_indices)
      {
        assert(child_count <= N);

        const BvhNode2& node = wdata.bvh2.nodes[node_idx];
        const Split& split = wdata.splits[node_idx * (N - 1) + child_idx];
        const Split& left_split = wdata.splits[node.field1 * (N - 1) + split.left_count];
        const Split& right_split = wdata.splits[node.field2 * (N - 1) + split.right_count];

        if (left_split.type == SplitType::Distribute)
        {
          collect_childs(wdata, node.field1, split.left_count, child_count, child_indices);
        }
        else
        {
          child_indices[child_count++] = node.field1;
        }

        if (right_split.type == SplitType::Distribute)
        {
          collect_childs(wdata, node.field2, split.right_count, child_count, child_indices);
        }
        else
        {
          child_indices[child_count++] = node.field2;
        }
      }

      template<size_t N>
      uint32_t push_leaves(const WorkData<N>& wdata,
                           uint32_t node_idx,
                           gml_aabb& aabb)
      {
        const BvhNode2& node = wdata.bvh2.nodes[node_idx];

        if ((node.field2 & 0x80000000) == 0x80000000)
        {
          gml_aabb_merge(&aabb, &node.aabb, &aabb);
          uint32_t face_count = (node.field2 & 0x7FFFFFFF);

          for (uint32_t i = 0; i < face_count; ++i)
          {
            gi_face face = wdata.bvh2.faces[node.field1 + i];
            wdata.bvh.faces.push_back(face);
          }

          return face_count;
        }

        return push_leaves(wdata, node.field1, aabb) + push_leaves(wdata, node.field2, aabb);
      }

      template<size_t N>
      uint32_t create_nodes(const WorkData<N>& wdata,
                            uint32_t node_idx,
                            BvhNode<N>& parent_node,
                            gml_aabb& parent_aabb)
      {
        // Inline nodes contained in distributed splits.
        uint32_t num_childs = 0;
        std::array<uint32_t, N> child_indices;
        collect_childs(wdata, node_idx, 0, num_childs, child_indices);

        // Create leaf nodes and internal node offsets.
        parent_node.child_index = wdata.bvh.nodes.size();
        parent_node.face_index = wdata.bvh.faces.size();

        for (uint32_t i = 0; i < num_childs; i++)
        {
          int32_t child_idx = child_indices[i];
          const Split& split = wdata.splits[child_idx * (N - 1)];

          if (split.type == SplitType::Leaf)
          {
            uint32_t face_offset = wdata.bvh.faces.size();
            uint32_t face_count = push_leaves(wdata, child_idx, parent_node.aabbs[i]);

            parent_node.offsets[i] = face_offset - parent_node.face_index;
            parent_node.counts[i] = (0x80000000 | face_count);

            gml_aabb_merge(&parent_aabb, &parent_node.aabbs[i], &parent_aabb);
          }
          else if (split.type == SplitType::Internal)
          {
            uint32_t new_idx = wdata.bvh.nodes.size();
            wdata.bvh.nodes.push_back(BvhNode<N>{});

            parent_node.offsets[i] = new_idx - parent_node.child_index;
          }
          else
          {
            assert(false);
          }
        }

        // Get internal node counts and AABBs by recursing into children.
        for (uint32_t i = 0; i < num_childs; i++)
        {
          int32_t child_idx = child_indices[i];
          const Split& split = wdata.splits[child_idx * (N - 1)];

          if (split.type != SplitType::Internal)
          {
            continue;
          }

          uint32_t new_idx = parent_node.child_index + parent_node.offsets[i];
          BvhNode<N>& new_node = wdata.bvh.nodes[new_idx];

          for (uint32_t k = 0; k < N; ++k)
          {
            new_node.counts[k] = 0;
            new_node.offsets[k] = 0;
            gml_aabb_make_smallest(&new_node.aabbs[k]);
          }

          parent_node.counts[i] = create_nodes(wdata, child_idx, new_node, parent_node.aabbs[i]);

          gml_aabb_merge(&parent_aabb, &parent_node.aabbs[i], &parent_aabb);
        }

        return num_childs;
      }

      template<size_t N>
      bool collapse_bvh2(const Bvh2& bvh2,
                         const CollapseParams& params,
                         Bvh<N>& bvh)
      {
        if (bvh2.faces.size() <= params.max_leaf_size)
        {
          // Too few triangles; we don't support leaf node as root.
          assert(false);
          return false;
        }

        // Calculate cost lookup table.
        uint32_t num_splits = bvh2.nodes.size() * (N - 1);

        WorkData<N> wdata(bvh2, bvh, params);
        wdata.splits.resize(num_splits);

        for (Split& split : wdata.splits)
        {
          split.type = SplitType::Invalid;
        }

        calc_costs(wdata, 0);

        // Set up the new BVH.
        bvh.aabb = bvh2.nodes[0].aabb;
        bvh.nodes.reserve(bvh2.nodes.size());
        bvh.faces.reserve(bvh2.faces.size());

        BvhNode<N> root_node;
        for (uint32_t j = 0; j < N; j++)
        {
          root_node.offsets[j] = 0;
          root_node.counts[j] = 0;
          gml_aabb_make_smallest(&root_node.aabbs[j]);
        }
        bvh.nodes.push_back(root_node);

        // Costruct wide BVH recursively with cost table.
        create_nodes(wdata, 0, bvh.nodes[0], bvh.aabb);

        // Since the leaves are collapsed, there are fewer of them.
        bvh.nodes.shrink_to_fit();

        return true;
      }
    }

    template<size_t N>
    bool collapse_bvh2(const Bvh2& bvh2,
                       const CollapseParams& params,
                       Bvh<N>& bvh)
    {
      static_assert(N > 2, "Collapsed BVH must be wider than 2");

      return detail::collapse_bvh2(bvh2, params, bvh);
    }
  }
}
