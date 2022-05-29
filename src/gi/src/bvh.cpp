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

#include "bvh.h"

#include "gi.h"

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <algorithm>

// This builder produces a binary BVH using the Surface Area Heuristic (SAH).
// It supports approximation through binning, as well as spatial splits. Memory
// is only allocated once at the beginning of the construction process. Subsequent
// memory accesses only happen into disjunct views of the working memory blob,
// which are called ranges.
// Each range represents a list of face references and has a stack pointer, a stack
// size, a direction and a size limit. With each split in the hierarchy, we divide
// a range into two smaller child ranges. This this done recursively until we
// reach the leaf layer. To accommodate for face duplications, we grow the child
// ranges inwards from the bounds of the parent range. A reserve buffer is used to
// allow spatial splitting for ranges with no free space left.
//
//   ┌───────────────────────────────────────────────────────────────────┐
//   │█████████████████████████████████                                  │  |
//   ├────────────────────────────────┬──────────────────────────────────┤  |
//   │████████████████                │                 █████████████████│  |
//   ├─────────────────┬──────────────┼────────────────────┬─────────────┤  |
//   │█████████        │       ███████│████████████        │        █████│  |
//   ├────────┬────────┼───────┬──────┼────┬───────────────┼───────┬─────┤  V
//   │█████   │    ████│████   │   ███│█   │    ███████████│███    │   ██│
//   └────────┴────────┴───────┴──────┴────┴───────────────┴───────┴─────┘
//
// It is based on these papers:
// - Ingo Wald (2007):
//   On fast Construction of SAH-based Bounding Volume Hierarchies
// - Martin Stich, Heiko Friedrich, and Andreas Dietrich (2009):
//   Spatial splits in bounding volume hierarchies
// - V. Fuetterling, C. Lojewski, F.-J. Pfreundt, and A. Ebert (2016):
//   Parallel spatial splits in bounding volume hierarchies

using namespace gi;
using namespace gi::bvh;

namespace impl
{
  struct FaceRef
  {
    gml_aabb aabb;
    uint32_t index;
  };

  struct ObjectBin
  {
    gml_aabb aabb;
    uint32_t face_count;
  };

  struct SpatialBin
  {
    uint32_t entry_count;
    uint32_t exit_count;
    gml_aabb aabb;
  };

  struct ObjectSplit
  {
    float sah_cost;
    uint32_t axis;
    float dcentroid;
    uint32_t face_index;
    float overlap_half_area;
  };

  struct BinnedObjectSplit
  {
    float sah_cost;
    uint32_t axis;
    uint32_t bin_index;
    float overlap_half_area;
  };

  struct SpatialSplit
  {
    float sah_cost;
    uint32_t axis;
    int32_t bin_index;
    uint32_t left_face_count;
    uint32_t right_face_count;
  };

  struct ThreadData
  {
    const BvhBuildParams* params;
    float root_half_area;
    void* reused_bins;
    gml_aabb* reused_aabbs;
  };

  using FaceRefIter = std::vector<FaceRef>::iterator;

  struct WorkRange
  {
    FaceRefIter stack;
    int32_t stack_dir;
    uint32_t stack_size;
    uint32_t stack_capacity;
    gml_aabb aabb;
    gml_aabb centroid_bounds;

    FaceRefIter start() { return stack_dir == 1 ? stack : (stack - (stack_size - 1)); }
    FaceRefIter end() { return start() + stack_size; }
    FaceRef& at(int index) const { assert(index >= 0 && index < stack_capacity); return stack[index * stack_dir]; }
  };

  struct WorkJob
  {
    WorkRange range;
    uint32_t node_index;
  };

  template<int AXIS>
  static bool sort_comp_func(const FaceRef& a, const FaceRef& b)
  {
    float aabb_dcentroid_a = a.aabb.min[AXIS] + a.aabb.max[AXIS];
    float aabb_dcentroid_b = b.aabb.min[AXIS] + b.aabb.max[AXIS];
    return (aabb_dcentroid_a < aabb_dcentroid_b) || (aabb_dcentroid_a == aabb_dcentroid_b && a.index < b.index);
  }

  static float face_test_cost(float base_cost, uint32_t batch_size, uint32_t face_count)
  {
    uint32_t rounded_to_batch_size = ((face_count + batch_size - 1) / batch_size) * batch_size;
    return rounded_to_batch_size * base_cost;
  }

  static void find_object_split(const ThreadData& thread_data, WorkRange& range, ObjectSplit& split)
  {
    float best_sah_cost = INFINITY;
    float best_tie_break = INFINITY;

    gml_aabb left_accum;
    gml_aabb right_accum;

    // Test each axis and sort faces along it.
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
      switch (axis)
      {
        case 0:
          std::sort(range.start(), range.end(), sort_comp_func<0>);
          break;
        case 1:
          std::sort(range.start(), range.end(), sort_comp_func<1>);
          break;
        case 2:
          std::sort(range.start(), range.end(), sort_comp_func<2>);
          break;
        default:
          assert(false);
      }

      // Sweep from right to left.
      gml_aabb_make_smallest(&right_accum);

      for (int32_t r = range.stack_size - 1; r > 0; --r)
      {
        const FaceRef& ref = range.start()[r];
        gml_aabb_merge(&right_accum, &ref.aabb, &right_accum);
        thread_data.reused_aabbs[r - 1] = right_accum;
      }

      // Sweep from left to right.
      gml_aabb_make_smallest(&left_accum);

      for (uint32_t l = 1; l < range.stack_size; ++l)
      {
        const FaceRef& ref = range.start()[(int32_t)l - 1];
        gml_aabb_merge(&left_accum, &ref.aabb, &left_accum);

        uint32_t r = range.stack_size - l;

        // Calculate SAH cost.
        float area_l = gml_aabb_half_area(&left_accum);
        float area_r = gml_aabb_half_area(&thread_data.reused_aabbs[l - 1]);

        float sah_cost =
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, l) * area_l +
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, r) * area_r;

        // Abort if cost is higher than best split.
        if (sah_cost > best_sah_cost)
        {
          continue;
        }

        // When SAH is equal, prefer equal face distribution.
        float tie_break = sqrtf((float)l) + sqrtf((float)r);

        if (sah_cost == best_sah_cost && tie_break > best_tie_break)
        {
          continue;
        }

        // Set new best split candidate.
        float dcentroid = ref.aabb.min[axis] + ref.aabb.max[axis];

        gml_aabb overlap_aabb;
        gml_aabb_intersect(&left_accum, &thread_data.reused_aabbs[l - 1], &overlap_aabb);

        split.sah_cost = sah_cost;
        split.axis = axis;
        split.dcentroid = dcentroid;
        split.face_index = ref.index;
        split.overlap_half_area = gml_aabb_half_area(&overlap_aabb);

        best_sah_cost = sah_cost;
        best_tie_break = tie_break;
      }
    }
  }

  static void find_binned_object_split(const ThreadData& thread_data,
                                       const WorkRange& range,
                                       BinnedObjectSplit& split)
  {
    float best_sah_cost = INFINITY;
    float best_tie_break = INFINITY;

    gml_aabb left_accum;
    gml_aabb right_accum;

    gml_vec3 axis_lengths;
    gml_vec3_sub(range.centroid_bounds.max, range.centroid_bounds.min, axis_lengths);

    uint32_t bin_count;
    if (thread_data.params->object_binning_mode == BvhBinningMode::Adaptive)
    {
      bin_count = i32min((uint32_t) (range.stack_size * 0.05f + 4.0f), thread_data.params->object_bin_count);
    }
    else
    {
      bin_count = thread_data.params->object_bin_count;
    }

    ObjectBin* bins = (ObjectBin*) thread_data.reused_bins;
    gml_aabb* reused_aabbs = (gml_aabb*) thread_data.reused_aabbs;

    // Test each axis.
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
      float axis_length = axis_lengths[axis];

      if (axis_length <= 0.0f)
      {
        continue;
      }

      float k1 = bin_count / axis_length;

      // Clear object bins.
      for (uint32_t i = 0; i < bin_count; ++i)
      {
        ObjectBin& bin = bins[i];
        gml_aabb_make_smallest(&bin.aabb);
        bin.face_count = 0;
      }

      // Project faces to bins.
      for (uint32_t i = 0; i < range.stack_size; ++i)
      {
        const FaceRef& ref = range.at(i);

        float centroid = (ref.aabb.min[axis] + ref.aabb.max[axis]) * 0.5f;

        int32_t bin_index = (int32_t) (k1 * (centroid - range.centroid_bounds.min[axis]));
        bin_index = i32max(0, i32min(bin_index, (bin_count - 1)));

        ObjectBin& bin = bins[bin_index];
        bin.face_count++;

        gml_aabb_merge(&bin.aabb, &ref.aabb, &bin.aabb);
      }

      // Sweep from right to left.
      gml_aabb_make_smallest(&right_accum);

      for (uint32_t r = bin_count - 1; r > 0; --r)
      {
        const ObjectBin& bin = bins[r];
        gml_aabb_merge(&right_accum, &bin.aabb, &right_accum);
        reused_aabbs[r - 1] = right_accum;
      }

      // Sweep from left to right.
      gml_aabb_make_smallest(&left_accum);

      uint32_t left_face_count = 0;

      for (uint32_t l = 1; l < bin_count; ++l)
      {
        const ObjectBin& bin = bins[l - 1];
        gml_aabb_merge(&left_accum, &bin.aabb, &left_accum);

        left_face_count += bin.face_count;
        uint32_t right_face_count = range.stack_size - left_face_count;

        // Calculate SAH cost.
        float area_l = gml_aabb_half_area(&left_accum);
        float area_r = gml_aabb_half_area(&reused_aabbs[l - 1]);

        float sah_cost =
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, left_face_count) * area_l +
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, right_face_count) * area_r;

        // Abort if cost is higher than best split.
        if (sah_cost > best_sah_cost)
        {
          continue;
        }

        // When SAH is equal, prefer equal face distribution.
        float tie_break = sqrtf((float)left_face_count) + sqrtf((float)right_face_count);

        if (sah_cost == best_sah_cost && tie_break > best_tie_break)
        {
          continue;
        }

        // Set new best split candidate.
        gml_aabb overlap_aabb;
        gml_aabb_intersect(&left_accum, &reused_aabbs[l - 1], &overlap_aabb);

        split.sah_cost = sah_cost;
        split.axis = axis;
        split.bin_index = l;
        split.overlap_half_area = gml_aabb_half_area(&overlap_aabb);

        best_sah_cost = sah_cost;
        best_tie_break = tie_break;
      }
    }
  }

  static void find_spatial_split(const ThreadData& thread_data,
                                 const WorkRange& range,
                                 SpatialSplit& split)
  {
    SpatialBin* bins = (SpatialBin*) thread_data.reused_bins;
    const gi_vertex* vertices = thread_data.params->vertices;
    const gi_face* faces = thread_data.params->faces;
    const gml_aabb& range_aabb = range.aabb;
    uint32_t bin_count = thread_data.params->spatial_bin_count;

    gml_vec3 axis_lengths;
    gml_aabb_size(&range.aabb, axis_lengths);
    gml_vec3 bin_sizes;
    gml_vec3_divs(axis_lengths, (float) bin_count, bin_sizes);

    // Clear spatial bins.
    for (uint32_t b = 0; b < bin_count * 3; ++b)
    {
      SpatialBin& bin = bins[b];
      bin.entry_count = 0;
      bin.exit_count = 0;
      gml_aabb_make_smallest(&bin.aabb);
    }

    // Fill spatial bins.
    for (uint32_t f = 0; f < range.stack_size; ++f)
    {
      const FaceRef& ref = range.at(f);

      for (uint32_t axis = 0; axis < 3; ++axis)
      {
        if (axis_lengths[axis] <= 0.0f)
        {
          continue;
        }

        float bin_size = bin_sizes[axis];

        const gml_aabb& ref_aabb = ref.aabb;
        const gi_face& face = faces[ref.index];

        gml_vec3 v_0;
        gml_vec3_assign(vertices[face.v_i[2]].pos, v_0);

        // Insert all three edges into bin AABBs.
        for (uint32_t e = 0; e < 3; ++e)
        {
          gml_vec3 v_1, v_start, v_end;
          gml_vec3_assign(vertices[face.v_i[e]].pos, v_1);
          gml_vec3_assign(v_0[axis] <= v_1[axis] ? v_0 : v_1, v_start);
          gml_vec3_assign(v_0[axis] <= v_1[axis] ? v_1 : v_0, v_end);
          gml_vec3_assign(v_1, v_0);

          if (v_start[axis] > range.aabb.max[axis] ||
              v_end[axis] < range.aabb.min[axis])
          {
            continue;
          }

          if (v_start[axis] < ref_aabb.min[axis])
          {
            float edge_length = v_end[axis] - v_start[axis];
            float t_plane_rel = (ref_aabb.min[axis] - v_start[axis]) / edge_length;
            gml_vec3_lerp(v_start, v_end, t_plane_rel, v_start);
            v_start[axis] = ref_aabb.min[axis];
          }
          if (v_end[axis] > ref_aabb.max[axis])
          {
            float edge_length = v_end[axis] - v_start[axis];
            float t_plane_rel = (ref_aabb.max[axis] - v_start[axis]) / edge_length;
            gml_vec3_lerp(v_start, v_end, t_plane_rel, v_end);
            v_end[axis] = ref_aabb.max[axis];
          }

          int32_t start_bin_index = (int32_t) ((v_start[axis] - range_aabb.min[axis]) / bin_size);
          int32_t end_bin_index = (int32_t) ((v_end[axis] - range_aabb.min[axis]) / bin_size);
          start_bin_index = i32max(0, i32min(start_bin_index, bin_count - 1));
          end_bin_index = i32max(0, i32min(end_bin_index, bin_count - 1));

          gml_aabb* start_bin_aabb = &bins[axis * bin_count + start_bin_index].aabb;
          gml_aabb* end_bin_aabb = &bins[axis * bin_count + end_bin_index].aabb;
          gml_aabb_include(start_bin_aabb, v_start, start_bin_aabb);
          gml_aabb_include(end_bin_aabb, v_end, end_bin_aabb);

          if (start_bin_index == end_bin_index)
          {
            continue;
          }

          // Include bin plane intersection points in both bin AABBs.
          for (int32_t bin_index = start_bin_index; bin_index < end_bin_index; ++bin_index)
          {
            float t_bin_end_plane = range_aabb.min[axis] + (float) (bin_index + 1) * bin_size;

            gml_vec3 v_i;
            float edge_length = v_end[axis] - v_start[axis];
            float t_plane_rel = (t_bin_end_plane - v_start[axis]) / edge_length;
            gml_vec3_lerp(v_start, v_end, t_plane_rel, v_i);
            v_i[axis] = t_bin_end_plane;

            SpatialBin& this_bin = bins[axis * bin_count + bin_index + 0];
            SpatialBin& next_bin = bins[axis * bin_count + bin_index + 1];
            gml_aabb_include(&this_bin.aabb, v_i, &this_bin.aabb);
            gml_aabb_include(&next_bin.aabb, v_i, &next_bin.aabb);
          }
        }

        // Increment entry and exit counters.
        int32_t start_bin_index = (int32_t) ((ref.aabb.min[axis] - range_aabb.min[axis]) / bin_size);
        int32_t end_bin_index = (int32_t) ((ref.aabb.max[axis] - range_aabb.min[axis]) / bin_size);
        start_bin_index = i32max(0, i32min(start_bin_index, bin_count - 1));
        end_bin_index = i32max(0, i32min(end_bin_index, bin_count - 1));

        bins[axis * bin_count + start_bin_index].entry_count++;
        bins[axis * bin_count + end_bin_index].exit_count++;
      }
    }

    // Evaluate split planes.
    float best_sah_cost = INFINITY;
    float best_tie_break = INFINITY;

    gml_aabb left_accum;
    gml_aabb right_accum;

    for (uint32_t axis = 0; axis < 3; ++axis)
    {
      if (axis_lengths[axis] <= 0.0f)
      {
        continue;
      }

      // Sweep from right to left.
      gml_aabb_make_smallest(&right_accum);

      for (int32_t r = bin_count - 1; r > 0; --r)
      {
        const SpatialBin& bin = bins[axis * bin_count + r];
        gml_aabb_merge(&right_accum, &bin.aabb, &right_accum);
        thread_data.reused_aabbs[r - 1] = right_accum;
      }

      // Sweep from left to right.
      gml_aabb_make_smallest(&left_accum);

      uint32_t left_face_count = 0;
      uint32_t right_face_count = range.stack_size;

      for (uint32_t l = 1; l < bin_count; ++l)
      {
        const SpatialBin& bin = bins[axis * bin_count + l - 1];
        gml_aabb_merge(&left_accum, &bin.aabb, &left_accum);

        left_face_count += bin.entry_count;
        right_face_count -= bin.exit_count;

        // Ignore invalid splits.
        if (left_face_count == 0 || right_face_count == 0)
        {
          continue;
        }

        // Calculate SAH cost.
        float area_l = gml_aabb_half_area(&left_accum);
        float area_r = gml_aabb_half_area(&thread_data.reused_aabbs[l - 1]);

        float sah_cost =
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, left_face_count) * area_l +
          face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, right_face_count) * area_r;

        // Abort if cost is higher than best split.
        if (sah_cost > best_sah_cost)
        {
          continue;
        }

        // When SAH is equal, prefer equal face distribution.
        float tie_break = sqrtf((float)left_face_count) + sqrtf((float)right_face_count);

        if (sah_cost == best_sah_cost && tie_break > best_tie_break)
        {
          continue;
        }

        // Set new best split candidate.
        split.sah_cost = sah_cost;
        split.axis = axis;
        split.bin_index = l;
        split.left_face_count = left_face_count;
        split.right_face_count = right_face_count;

        best_sah_cost = sah_cost;
        best_tie_break = tie_break;
      }
    }
  }

  static void do_spatial_split(const ThreadData& thread_data,
                               const SpatialSplit& split,
                               const WorkRange& range,
                               WorkRange& range_left,
                               WorkRange& range_right)
  {
    uint32_t bin_count = thread_data.params->spatial_bin_count;
    float axis_length = range.aabb.max[split.axis] - range.aabb.min[split.axis];
    float bin_size = axis_length / bin_count;

    const gi_vertex* vertices = thread_data.params->vertices;
    const gi_face* faces = thread_data.params->faces;
    uint32_t axis = split.axis;

    int32_t split_face_count = split.left_face_count + split.right_face_count;
    int32_t free_face_count = range.stack_capacity - split_face_count;
    assert(free_face_count >= 0);

    WorkRange& range1 = range.stack_dir == 1 ? range_left : range_right;
    WorkRange& range2 = range.stack_dir == 1 ? range_right : range_left;

    int32_t range1_index_start = 0;
    int32_t range1_index_end = range.stack_size - 1;
    int32_t range2_index_start = range.stack_capacity - 1;

    range1.stack_size = 0;
    range1.stack_dir = range.stack_dir;
    range1.stack = range.stack;
    gml_aabb_make_smallest(&range1.aabb);
    gml_aabb_make_smallest(&range1.centroid_bounds);

    range2.stack_size = 0;
    range2.stack_dir = range.stack_dir * -1;
    range2.stack = range.stack + range2_index_start * range.stack_dir;
    gml_aabb_make_smallest(&range2.aabb);
    gml_aabb_make_smallest(&range2.centroid_bounds);

    while (range1_index_start <= range1_index_end)
    {
      // Note that this reference is not const! It will be changed.
      FaceRef& ref = range.at(range1_index_start);

      const gml_aabb& ref_aabb = ref.aabb;
      const gi_face& face = faces[ref.index];

      // Split all edges on the split plane and get AABBs for both sides.

      gml_aabb left_aabb;
      gml_aabb right_aabb;
      gml_aabb_make_smallest(&left_aabb);
      gml_aabb_make_smallest(&right_aabb);

      gml_vec3 v_0;
      gml_vec3_assign(vertices[face.v_i[2]].pos, v_0);

      for (uint32_t e = 0; e < 3; ++e)
      {
        gml_vec3 v_1, v_start, v_end;
        gml_vec3_assign(vertices[face.v_i[e]].pos, v_1);
        gml_vec3_assign(v_0[split.axis] <= v_1[split.axis] ? v_0 : v_1, v_start);
        gml_vec3_assign(v_0[split.axis] <= v_1[split.axis] ? v_1 : v_0, v_end);
        gml_vec3_assign(v_1, v_0);

        // Cull and chop edge.

        if (v_start[split.axis] > range.aabb.max[split.axis] ||
            v_end[split.axis] < range.aabb.min[split.axis])
        {
          continue;
        }

        if (v_start[axis] < ref_aabb.min[axis])
        {
          float edge_length = v_end[axis] - v_start[axis];
          float t_plane_rel = (ref_aabb.min[axis] - v_start[axis]) / edge_length;
          gml_vec3_lerp(v_start, v_end, t_plane_rel, v_start);
          v_start[axis] = ref_aabb.min[axis];
        }
        if (v_end[axis] > ref_aabb.max[axis])
        {
          float edge_length = v_end[axis] - v_start[axis];
          float t_plane_rel = (ref_aabb.max[axis] - v_start[axis]) / edge_length;
          gml_vec3_lerp(v_start, v_end, t_plane_rel, v_end);
          v_end[axis] = ref_aabb.max[axis];
        }

        // Fill left and right AABBs.

        float t_plane = range.aabb.min[axis] + split.bin_index * bin_size;

        if (v_start[axis] <= t_plane) { gml_aabb_include(&left_aabb, v_start, &left_aabb); }
        if (v_start[axis] >= t_plane) { gml_aabb_include(&right_aabb, v_start, &right_aabb); }
        if (v_end[axis] <= t_plane) { gml_aabb_include(&left_aabb, v_end, &left_aabb); }
        if (v_end[axis] >= t_plane) { gml_aabb_include(&right_aabb, v_end, &right_aabb); }

        // Continue if there is no plane intersection.
        if (t_plane < v_start[axis] || t_plane > v_end[axis] ||
            (t_plane == v_start[axis] && t_plane == v_end[axis]))
        {
          continue;
        }

        // Otherwise, split into two halves.

        float edge_length = v_end[axis] - v_start[axis];
        float t_plane_abs = t_plane - v_start[axis];
        float t_plane_rel = (t_plane_abs / edge_length);

        gml_vec3 v_i;
        gml_vec3_lerp(v_start, v_end, t_plane_rel, v_i);
        v_i[axis] = t_plane;

        gml_aabb_include(&left_aabb, v_i, &left_aabb);
        gml_aabb_include(&right_aabb, v_i, &right_aabb);
      }

      gml_aabb_intersect(&left_aabb, &ref.aabb, &left_aabb);
      gml_aabb_intersect(&right_aabb, &ref.aabb, &right_aabb);

      // Now that we have both side AABBs, we do the actual partitioning.

      int32_t start_bin_index = (int32_t) ((ref.aabb.min[axis] - range.aabb.min[axis]) / bin_size);
      int32_t end_bin_index = (int32_t) ((ref.aabb.max[axis] - range.aabb.min[axis]) / bin_size);
      start_bin_index = i32max(0, i32min(start_bin_index, bin_count - 1));
      end_bin_index = i32max(0, i32min(end_bin_index, bin_count - 1));

      bool is_in_left = start_bin_index < split.bin_index;
      bool is_in_right = end_bin_index >= split.bin_index;

      bool is_in_range1 = (range.stack_dir == 1 && is_in_left) || (range.stack_dir == -1 && is_in_right);
      bool is_in_range2 = (range.stack_dir == 1 && is_in_right) || (range.stack_dir == -1 && is_in_left);
      assert(is_in_range1 || is_in_range2);

      const gml_aabb& new_ref_aabb_r1 = (range.stack_dir == 1) ? left_aabb : right_aabb;
      const gml_aabb& new_ref_aabb_r2 = (range.stack_dir == 1) ? right_aabb : left_aabb;

      // Handle face being in the far range.

      if (is_in_range2)
      {
        range2.stack_size++;

        // Check if there is space left or if we need to swap.
        if (range2_index_start != range1_index_end)
        {
          range.at(range2_index_start).index = ref.index;
          range.at(range2_index_start).aabb = new_ref_aabb_r2;

          // If the face is not duplicated, pull the next one.
          if (!is_in_range1)
          {
            range.at(range1_index_start) = range.at(range1_index_end);
            range1_index_end--;
          }
        }
        else
        {
          // Swap faces and overwrite our AABB with the new, chopped one.
          FaceRef tmp = range.at(range2_index_start);
          range.at(range2_index_start).index = ref.index;
          range.at(range2_index_start).aabb = new_ref_aabb_r2;
          range.at(range1_index_start) = tmp;
          range1_index_end--;
        }

        const FaceRef& new_ref = range.at(range2_index_start);

        gml_aabb_merge(&range2.aabb, &new_ref.aabb, &range2.aabb);

        gml_vec3 new_centroid;
        gml_vec3_add(new_ref.aabb.min, new_ref.aabb.max, new_centroid);
        gml_vec3_muls(new_centroid, 0.5f, new_centroid);
        gml_aabb_include(&range2.centroid_bounds, new_centroid, &range2.centroid_bounds);

        range2_index_start--;
      }

      // Handle face being in the close range.

      if (is_in_range1)
      {
        range1.stack_size++;

        ref.aabb = new_ref_aabb_r1;

        gml_aabb_merge(&range1.aabb, &ref.aabb, &range1.aabb);

        gml_vec3 new_centroid;
        gml_vec3_add(ref.aabb.min, ref.aabb.max, new_centroid);
        gml_vec3_muls(new_centroid, 0.5f, new_centroid);
        gml_aabb_include(&range1.centroid_bounds, new_centroid, &range1.centroid_bounds);

        range1_index_start++;
      }
    }

    range1.stack_capacity = range1.stack_size + free_face_count / 2;
    range2.stack_capacity = range2.stack_size + free_face_count - free_face_count / 2;

    assert(range_left.stack_size == split.left_face_count);
    assert(range_right.stack_size == split.right_face_count);
  }

  static void do_object_split(const ObjectSplit& split,
                              const WorkRange& range,
                              WorkRange& range_left,
                              WorkRange& range_right)
  {
    // This partitioning algorithm is a little bit special since we
    // deal with directional ranges. While partitioning, we want both
    // sides to grow inwards from the range bounds. This is performed
    // in-place by taking into account the availability of free space.
    // Left and right face counts are commonly used for partitioning,
    // but because we don't sort in the first place, we can only rely
    // on the centroid position. In case of ambiguities, we compare
    // the referenced face indices. These are unique within a range.
    //
    // Standard partitioning schemes:
    // ┌──────────────────┐    ┌──────────────────┐
    // │X█XX██X██         │ -> │XXXX█████         │
    // └──────────────────┘    └──────────────────┘
    // Our partitioning scheme:
    // ┌──────────────────┐    ┌──────────────────┐
    // │X█XX██X██         │ -> │XXXX         █████│
    // └──────────────────┘    └──────────────────┘

    // Range 1 is the side close to the origin of the parent stack.
    // Range 2 represents the opposite side.

    WorkRange& range1 = range.stack_dir == 1 ? range_left : range_right;
    WorkRange& range2 = range.stack_dir == 1 ? range_right : range_left;

    int32_t range1_index_start = 0;
    int32_t range1_index_end = range.stack_size - 1;
    int32_t range2_index_start = range.stack_capacity - 1;

    // Reset child ranges.

    range1.stack_dir = range.stack_dir;
    range1.stack = range.stack;
    range1.stack_size = 0;
    gml_aabb_make_smallest(&range1.aabb);
    gml_aabb_make_smallest(&range1.centroid_bounds);

    range2.stack_dir = range.stack_dir * -1;
    range2.stack = range.stack + range.stack_dir * range2_index_start;
    range2.stack_size = 0;
    gml_aabb_make_smallest(&range2.aabb);
    gml_aabb_make_smallest(&range2.centroid_bounds);

    // Do partitioning.

    bool stack_dir_pos = range.stack_dir == +1;
    bool stack_dir_neg = range.stack_dir == -1;

    while (range1_index_start <= range1_index_end)
    {
      const FaceRef& ref = range.at(range1_index_start);

      gml_vec3 centroid;
      gml_vec3_add(ref.aabb.min, ref.aabb.max, centroid);

      bool is_in_left = (centroid[split.axis] < split.dcentroid) ||
                        (centroid[split.axis] == split.dcentroid && ref.index <= split.face_index);

      bool is_in_range1 = (stack_dir_pos && is_in_left) || (stack_dir_neg && !is_in_left);

      gml_vec3_muls(centroid, 0.5f, centroid);

      // Handle face being in the close range.

      if (is_in_range1)
      {
        range1.stack_size++;

        gml_aabb_include(&range1.centroid_bounds, centroid, &range1.centroid_bounds);
        gml_aabb_merge(&range1.aabb, &ref.aabb, &range1.aabb);

        range1_index_start++;
        continue;
      }

      // Handle face being in the far range.

      range2.stack_size++;

      gml_aabb_include(&range2.centroid_bounds, centroid, &range2.centroid_bounds);
      gml_aabb_merge(&range2.aabb, &ref.aabb, &range2.aabb);

      // Check if there is space left or if we need to swap.
      if (range2_index_start != range1_index_end)
      {
        range.at(range2_index_start) = ref;
        range.at(range1_index_start) = range.at(range1_index_end);
      }
      else
      {
        FaceRef tmp = range.at(range2_index_start);
        range.at(range2_index_start) = range.at(range1_index_start);
        range.at(range1_index_start) = tmp;
      }

      range2_index_start--;
      range1_index_end--;
    }

    assert(range1.stack_size > 0);
    assert(range2.stack_size > 0);

    // Assign stack limits.

    int32_t free_face_count = range.stack_capacity - range.stack_size;
    int32_t half_free_face_count = free_face_count / 2;
    range1.stack_capacity = range1.stack_size + half_free_face_count;
    range2.stack_capacity = range2.stack_size + (free_face_count - half_free_face_count);
  }

  static void do_binned_object_split(const ThreadData& thread_data,
                                     const BinnedObjectSplit& split,
                                     const WorkRange& range,
                                     WorkRange& range_left,
                                     WorkRange& range_right)
  {
    // See non-binned object splitting for a general algorithm description.

    WorkRange& range1 = range.stack_dir == 1 ? range_left : range_right;
    WorkRange& range2 = range.stack_dir == 1 ? range_right : range_left;

    int32_t range1_index_start = 0;
    int32_t range1_index_end = range.stack_size - 1;
    int32_t range2_index_start = range.stack_capacity - 1;

    // Reset child ranges.

    range1.stack_dir = range.stack_dir;
    range1.stack = range.stack;
    range1.stack_size = 0;
    gml_aabb_make_smallest(&range1.aabb);
    gml_aabb_make_smallest(&range1.centroid_bounds);

    range2.stack_dir = range.stack_dir * -1;
    range2.stack = range.stack + range.stack_dir * range2_index_start;
    range2.stack_size = 0;
    gml_aabb_make_smallest(&range2.aabb);
    gml_aabb_make_smallest(&range2.centroid_bounds);

    // Precalculate values.

    float axis_length = range.centroid_bounds.max[split.axis] - range.centroid_bounds.min[split.axis];

    uint32_t bin_count;
    if (thread_data.params->object_binning_mode == BvhBinningMode::Adaptive)
    {
      bin_count = i32min((uint32_t) (range.stack_size * 0.05f + 4.0f), thread_data.params->object_bin_count);
    }
    else
    {
      bin_count = thread_data.params->object_bin_count;
    }

    float k1 = bin_count / axis_length;
    bool stack_dir_pos = range.stack_dir == +1;
    bool stack_dir_neg = range.stack_dir == -1;

    // Do partitioning.

    while (range1_index_start <= range1_index_end)
    {
      // Determining which side a face is on is done by re-projecting it
      // to its bin and comparing the bin index to the split bin index.
      // This should be consistent with the way we found the split - this
      // way, we don't have any deviations due to floating point math.

      const FaceRef& ref = range.at(range1_index_start);

      gml_vec3 centroid;
      gml_vec3_add(ref.aabb.min, ref.aabb.max, centroid);
      gml_vec3_muls(centroid, 0.5f, centroid);

      int32_t bin_index = (int32_t) (k1 * (centroid[split.axis] - range.centroid_bounds.min[split.axis]));
      bin_index = i32max(0, i32min(bin_index, (bin_count - 1)));

      bool is_in_range1 = (stack_dir_pos && bin_index < split.bin_index) ||
                          (stack_dir_neg && bin_index >= split.bin_index);

      // Handle face being in the close range. This partitioning algorithm
      // is essentially the same as in the non-binned object split.

      if (is_in_range1)
      {
        range1.stack_size++;

        gml_aabb_include(&range1.centroid_bounds, centroid, &range1.centroid_bounds);
        gml_aabb_merge(&range1.aabb, &ref.aabb, &range1.aabb);

        range1_index_start++;
        continue;
      }

      // Handle face being in the far range.

      range2.stack_size++;

      gml_aabb_include(&range2.centroid_bounds, centroid, &range2.centroid_bounds);
      gml_aabb_merge(&range2.aabb, &ref.aabb, &range2.aabb);

      // Check if there is space left or if we need to swap.
      if (range2_index_start != range1_index_end)
      {
        range.at(range2_index_start) = ref;
        range.at(range1_index_start) = range.at(range1_index_end);
      }
      else
      {
        FaceRef tmp = range.at(range2_index_start);
        range.at(range2_index_start) = range.at(range1_index_start);
        range.at(range1_index_start) = tmp;
      }

      range2_index_start--;
      range1_index_end--;
    }

    assert(range1.stack_size > 0);
    assert(range2.stack_size > 0);

    // Assign stack limits.

    int32_t free_face_count = range.stack_capacity - range.stack_size;
    int32_t half_free_face_count = free_face_count / 2;
    range1.stack_capacity = range1.stack_size + half_free_face_count;
    range2.stack_capacity = range2.stack_size + (free_face_count - half_free_face_count);
  }

  static bool build_work_range(const ThreadData& thread_data,
                               WorkRange& range,
                               WorkRange& range_left,
                               WorkRange& range_right)
  {
    // Make a leaf if face count is too low.
    if (range.stack_size == 1)
    {
      return false;
    }

    // Check if we want to use binning.
    bool is_cb_degenerate;
    {
      gml_vec3 cbmin, cbmax;
      gml_vec3_assign(range.centroid_bounds.min, cbmin);
      gml_vec3_assign(range.centroid_bounds.max, cbmax);

      is_cb_degenerate = (cbmin[0] == cbmax[0] && cbmin[1] == cbmax[1]) ||
                         (cbmin[1] == cbmax[1] && cbmin[2] == cbmax[2]) ||
                         (cbmin[2] == cbmax[2] && cbmin[0] == cbmax[0]);
    }

    bool should_use_binning = range.stack_size > thread_data.params->object_binning_threshold;
    bool is_binning_enabled = thread_data.params->object_binning_mode != BvhBinningMode::Off;

    bool do_binning = is_binning_enabled && !is_cb_degenerate && should_use_binning;

    // Evaluate possible splits.
    BinnedObjectSplit split_object_binned;
    ObjectSplit split_object;
    SpatialSplit split_spatial;

    split_object_binned.sah_cost = INFINITY;
    split_object_binned.bin_index = 0;
    split_object_binned.overlap_half_area = 0.0f;
    split_object.sah_cost = INFINITY;
    split_object.overlap_half_area = 0.0f;
    split_spatial.sah_cost = INFINITY;

    float overlap_half_area;

    if (do_binning)
    {
      find_binned_object_split(thread_data, range, split_object_binned);
      overlap_half_area = split_object_binned.overlap_half_area;
    }
    else
    {
      find_object_split(thread_data, range, split_object);
      overlap_half_area = split_object.overlap_half_area;
    }

    bool try_spatial_split = (overlap_half_area / thread_data.root_half_area) > thread_data.params->spatial_split_alpha;

    if (try_spatial_split)
    {
      find_spatial_split(thread_data, range, split_spatial);
    }

    float leaf_sah_cost = gml_aabb_half_area(&range.aabb) *
      face_test_cost(thread_data.params->face_intersection_cost, thread_data.params->face_batch_size, range.stack_size);

    // Find best split option.
    float best_sah_cost = fminf(
      fminf(split_object.sah_cost, split_object_binned.sah_cost),
      fminf(split_spatial.sah_cost, leaf_sah_cost)
    );

    // Handle best split option.
    bool fits_in_leaf = range.stack_size <= thread_data.params->leaf_max_face_count;

    if (fits_in_leaf && best_sah_cost == leaf_sah_cost)
    {
      return false;
    }

    if (best_sah_cost == split_spatial.sah_cost)
    {
      int32_t split_face_count = split_spatial.left_face_count + split_spatial.right_face_count;
      int32_t free_face_count = ((int32_t) range.stack_capacity) - split_face_count;

      if (free_face_count >= 0)
      {
        do_spatial_split(thread_data, split_spatial, range, range_left, range_right);
        return true;
      }
    }

    if (do_binning)
    {
      do_binned_object_split(thread_data, split_object_binned, range, range_left, range_right);
    }
    else
    {
      do_object_split(split_object, range, range_left, range_right);
    }

    return true;
  }

  Bvh2 build_bvh2(const BvhBuildParams& params)
  {
    // Determine root AABB and remove degenerate faces.
    gml_aabb root_aabb;
    gml_aabb root_centroid_bounds;
    gml_aabb_make_smallest(&root_aabb);
    gml_aabb_make_smallest(&root_centroid_bounds);

    uint32_t root_stack_capacity = params.face_count;
    if (params.spatial_split_alpha < 1.0f)
    {
      root_stack_capacity *= 2;
    }

    std::vector<FaceRef> root_stack;
    root_stack.resize(root_stack_capacity);

    uint32_t root_stack_size = 0;
    for (uint32_t i = 0; i < params.face_count; ++i)
    {
      const gi_face& face = params.faces[i];
      const gi_vertex& v_a = params.vertices[face.v_i[0]];
      const gi_vertex& v_b = params.vertices[face.v_i[1]];
      const gi_vertex& v_c = params.vertices[face.v_i[2]];

      FaceRef& face_ref = root_stack[root_stack_size];

      gml_aabb_make_from_triangle(v_a.pos, v_b.pos, v_c.pos, &face_ref.aabb);

      bool is_one_dimensional =
        (face_ref.aabb.min[0] == face_ref.aabb.max[0] && face_ref.aabb.min[1] == face_ref.aabb.max[1]) ||
        (face_ref.aabb.min[1] == face_ref.aabb.max[1] && face_ref.aabb.min[2] == face_ref.aabb.max[2]) ||
        (face_ref.aabb.min[2] == face_ref.aabb.max[2] && face_ref.aabb.min[0] == face_ref.aabb.max[0]);

      if (is_one_dimensional)
      {
        continue;
      }

      face_ref.index = i;

      gml_aabb_merge(&root_aabb, &face_ref.aabb, &root_aabb);

      gml_vec3 centroid;
      gml_vec3_add(face_ref.aabb.max, face_ref.aabb.min, centroid);
      gml_vec3_muls(centroid, 0.5f, centroid);

      gml_aabb_include(&root_centroid_bounds, centroid, &root_centroid_bounds);

      root_stack_size++;
    }

    float root_half_area = gml_aabb_half_area(&root_aabb);

    // Set up bvh.
    uint32_t max_face_count = (params.spatial_split_alpha < 1.0f) ? params.face_count * 8 : params.face_count;
    uint32_t max_node_count = max_face_count * 2;

    Bvh2 bvh;
    bvh.faces.reserve(max_face_count);
    bvh.nodes.reserve(max_node_count);
    bvh.nodes.push_back({}); // Root node

    // Set up job pool.
    WorkJob root_job;
    root_job.range.stack            = root_stack.begin();
    root_job.range.stack_dir        = 1;
    root_job.range.stack_size       = root_stack_size;
    root_job.range.stack_capacity   = root_stack_capacity;
    root_job.range.aabb      = root_aabb;
    root_job.range.centroid_bounds  = root_centroid_bounds;
    root_job.node_index             = 0;

    std::vector<WorkJob> job_stack;
    job_stack.reserve(params.face_count);
    job_stack.push_back(root_job);

    // Allocate thread-local memory.
    uint32_t object_bins_size = params.object_bin_count * sizeof(ObjectBin);
    uint32_t spatial_bins_size = (params.spatial_split_alpha == 1.0f) ? 0 : (params.spatial_bin_count * sizeof(SpatialBin) * 3);
    uint32_t reused_bins_size = imax(object_bins_size, spatial_bins_size);
    int32_t reused_aabbs_size = imax(params.spatial_bin_count, params.face_count) * sizeof(gml_aabb);

    ThreadData thread_data;
    thread_data.params = &params;
    thread_data.reused_bins = (void*) malloc(reused_bins_size);
    thread_data.reused_aabbs = (gml_aabb*) malloc(reused_aabbs_size);
    thread_data.root_half_area = root_half_area;

    // Build the BVH (without allocating any memory)
    while (!job_stack.empty())
    {
      WorkJob job = job_stack.back();
      job_stack.pop_back();

      WorkRange left_range;
      WorkRange right_range;

      bool make_leaf = !build_work_range(thread_data, job.range, left_range, right_range);

      BvhNode2& node = bvh.nodes[job.node_index];

      node.aabb = job.range.aabb;

      // We did not split the range, make a leaf instead.
      if (make_leaf)
      {
        node.field1 = bvh.faces.size();
        node.field2 = (0x80000000 | job.range.stack_size);

        for (uint32_t i = 0; i < job.range.stack_size; ++i)
        {
          const FaceRef& ref = job.range.at(i);
          const gi_face& face = params.faces[ref.index];
          bvh.faces.push_back(face);
        }

        continue;
      }

      // Otherwise, create two child nodes.
      node.field1 = bvh.nodes.size();
      bvh.nodes.push_back({});
      node.field2 = bvh.nodes.size();
      bvh.nodes.push_back({});

      // And enqueue them for processing.
      WorkJob left_job;
      left_job.range = left_range;
      left_job.node_index = node.field1;
      job_stack.push_back(left_job);

      WorkJob right_job;
      right_job.range = right_range;
      right_job.node_index = node.field2;
      job_stack.push_back(right_job);
    }

    // Free memory.
    free(thread_data.reused_bins);
    free(thread_data.reused_aabbs);

    // Reduce memory usage.
    bvh.nodes.shrink_to_fit();
    bvh.faces.shrink_to_fit();

    return bvh;
  }
}

Bvh2 gi::bvh::build_bvh2(const BvhBuildParams& params)
{
  return impl::build_bvh2(params);
}
