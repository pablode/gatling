#include "bvh.h"

#include "gi.h"

#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

/*
 * This builder produces a binary BVH using the Surface Area Heuristic (SAH).
 * It supports approximation through binning, as well as spatial splits. Memory
 * is only allocated once at the beginning of the construction process. Subsequent
 * memory accesses only happen into disjunct views of the working memory blob,
 * which are called ranges.
 * Each range represents a list of face references and has a stack pointer, a stack
 * size, a direction and a size limit. With each split in the hierarchy, we divide
 * a range into two smaller child ranges. This this done recursively until we
 * reach the leaf layer. To accommodate for face duplications, we grow the child
 * ranges inwards from the bounds of the parent range. A reserve buffer is used to
 * allow spatial splitting for ranges with no free space left.
 *
 *   ┌───────────────────────────────────────────────────────────────────┐
 *   │█████████████████████████████████                                  │
 *   ├────────────────────────────────┬──────────────────────────────────┤
 *   │████████████████                │                 █████████████████│
 *   ├─────────────────┬──────────────┼────────────────────┬─────────────┤
 *   │█████████        │       ███████│████████████        │        █████│
 *   ├────────┬────────┼───────┬──────┼────┬───────────────┼───────┬─────┤
 *   │█████   │    ████│████   │   ███│█   │    ███████████│███    │   ██│
 *   └────────┴────────┴───────┴──────┴────┴───────────────┴───────┴─────┘
 *
 * Literature:
 *   - Ingo Wald. 2007.
 *     On fast Construction of SAH-based Bounding Volume Hierarchies.
 *     In Proceedings of the 2007 IEEE Symposium on Interactive Ray Tracing (RT ’07).
 *     IEEE Computer Society, USA, 33–40.
 *     DOI: https://doi.org/10.1109/RT.2007.4342588
 *   - Martin Stich, Heiko Friedrich, and Andreas Dietrich. 2009.
 *     Spatial splits in bounding volume hierarchies.
 *     In Proceedings of the Conference on High Performance Graphics 2009 (HPG ’09).
 *     Association for Computing Machinery, New York, NY, USA, 7–13.
 *     DOI: https://doi.org/10.1145/1572769.1572771
 *   - V. Fuetterling, C. Lojewski, F.-J. Pfreundt, and A. Ebert. 2016.
 *     Parallel spatial splits in bounding volume hierarchies.
 *     In Proceedings of the 16th Eurographics Symposium on Parallel Graphics and Visualization (EGPGV ’16).
 *     Eurographics Association, Goslar, DEU, 21–30.
 *     DOI: https://doi.org/10.2312/pgv.20161179
 */

struct gi_bvh_face_ref
{
  gml_aabb aabb;
  uint32_t index;
};

struct gi_bvh_object_bin
{
  gml_aabb aabb;
  uint32_t face_count;
};

struct gi_bvh_spatial_bin
{
  uint32_t entry_count;
  uint32_t exit_count;
  gml_aabb aabb;
};

struct gi_bvh_split_object
{
  float sah_cost;
  uint32_t axis;
  float dcentroid;
  uint32_t face_index;
  float overlap_half_area;
};

struct gi_bvh_split_object_binned
{
  float sah_cost;
  uint32_t axis;
  uint32_t bin_index;
  float overlap_half_area;
};

struct gi_bvh_split_spatial
{
  float sah_cost;
  uint32_t axis;
  int32_t bin_index;
  uint32_t left_face_count;
  uint32_t right_face_count;
};

struct gi_bvh_thread_data
{
  const struct gi_bvh_params* params;
  float root_half_area;
  void* reused_bins;
  gml_aabb* reused_aabbs;
  struct gi_bvh_face_ref* reserve_buffer;
  int32_t* reserve_buffer_capacity;
};

struct gi_bvh_work_range
{
  struct gi_bvh_face_ref* stack;
  int32_t stack_dir;
  uint32_t stack_size;
  uint32_t stack_size_limit;
  gml_aabb aabb_bounds;
  gml_aabb centroid_bounds;
};

struct gi_bvh_work_job
{
  struct gi_bvh_work_range range;
  uint32_t node_index;
};

#define GI_BVH_GEN_SORT_CMP_FUNC(AXIS)                                            \
  static int gi_bvh_sort_comp_func_##AXIS(                                        \
    const void* a, const void* b)                                                 \
  {                                                                               \
    const struct gi_bvh_face_ref* ref_1 = (struct gi_bvh_face_ref*) a;            \
    const struct gi_bvh_face_ref* ref_2 = (struct gi_bvh_face_ref*) b;            \
    float aabb_dcentroid_1 = ref_1->aabb.min[AXIS] + ref_1->aabb.max[AXIS];       \
    float aabb_dcentroid_2 = ref_2->aabb.min[AXIS] + ref_2->aabb.max[AXIS];       \
                                                                                  \
    if      (aabb_dcentroid_1 > aabb_dcentroid_2) { return  1; }                  \
    else if (aabb_dcentroid_1 < aabb_dcentroid_2) { return -1; }                  \
    else if (ref_1->index > ref_2->index)         { return  1; }                  \
    else if (ref_1->index < ref_2->index)         { return -1; }                  \
    else {                                                                        \
      return 0;                                                                   \
    }                                                                             \
  }

GI_BVH_GEN_SORT_CMP_FUNC(0)
GI_BVH_GEN_SORT_CMP_FUNC(1)
GI_BVH_GEN_SORT_CMP_FUNC(2)

static float gi_bvh_face_test_cost(float base_cost,
                                   uint32_t batch_size,
                                   uint32_t face_count)
{
  uint32_t rounded_to_batch_size = ((face_count + batch_size - 1) / batch_size) * batch_size;
  return rounded_to_batch_size * base_cost;
}

static void gi_bvh_find_split_object(const struct gi_bvh_thread_data* thread_data,
                                     const struct gi_bvh_work_range* range,
                                     struct gi_bvh_split_object* split)
{
  float best_sah_cost = INFINITY;
  float best_tie_break = INFINITY;

  gml_aabb left_accum;
  gml_aabb right_accum;

  struct gi_bvh_face_ref* range_stack_left = (range->stack_dir == 1) ? range->stack : (range->stack - (range->stack_size - 1));

  /* Test each axis and sort faces along it. */
  for (uint32_t axis = 0; axis < 3; ++axis)
  {
    switch (axis)
    {
      case 0:
        qsort(range_stack_left, range->stack_size, sizeof(struct gi_bvh_face_ref), gi_bvh_sort_comp_func_0);
        break;
      case 1:
        qsort(range_stack_left, range->stack_size, sizeof(struct gi_bvh_face_ref), gi_bvh_sort_comp_func_1);
        break;
      case 2:
        qsort(range_stack_left, range->stack_size, sizeof(struct gi_bvh_face_ref), gi_bvh_sort_comp_func_2);
        break;
      default:
        assert(false);
    }

    /* Sweep from right to left. */
    gml_aabb_make_smallest(&right_accum);

    for (int32_t r = range->stack_size - 1; r > 0; --r)
    {
      const struct gi_bvh_face_ref* ref = &range_stack_left[r];
      gml_aabb_merge(&right_accum, &ref->aabb, &right_accum);
      thread_data->reused_aabbs[r - 1] = right_accum;
    }

    /* Sweep from left to right. */
    gml_aabb_make_smallest(&left_accum);

    for (uint32_t l = 1; l < range->stack_size; ++l)
    {
      const struct gi_bvh_face_ref* ref = &range_stack_left[(int32_t)l - 1];
      gml_aabb_merge(&left_accum, &ref->aabb, &left_accum);

      uint32_t r = range->stack_size - l;

      /* Calculate SAH cost. */
      float area_l = gml_aabb_half_area(&left_accum);
      float area_r = gml_aabb_half_area(&thread_data->reused_aabbs[l - 1]);

      float sah_cost =
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, l) * area_l +
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, r) * area_r;

      /* Abort if cost is higher than best split. */
      if (sah_cost > best_sah_cost)
      {
        continue;
      }

      /* When SAH is equal, prefer equal face distribution. */
      float tie_break = sqrtf((float)l) + sqrtf((float)r);

      if (sah_cost == best_sah_cost && tie_break > best_tie_break)
      {
        continue;
      }

      /* Set new best split candidate. */
      float dcentroid = ref->aabb.min[axis] + ref->aabb.max[axis];

      gml_aabb overlap_aabb;
      gml_aabb_intersect(&left_accum, &thread_data->reused_aabbs[l - 1], &overlap_aabb);

      split->sah_cost = sah_cost;
      split->axis = axis;
      split->dcentroid = dcentroid;
      split->face_index = ref->index;
      split->overlap_half_area = gml_aabb_half_area(&overlap_aabb);

      best_sah_cost = sah_cost;
      best_tie_break = tie_break;
    }
  }
}

static void gi_bvh_find_split_object_binned(const struct gi_bvh_thread_data* thread_data,
                                            const struct gi_bvh_work_range* range,
                                            struct gi_bvh_split_object_binned* split)
{
  float best_sah_cost = INFINITY;
  float best_tie_break = INFINITY;

  gml_aabb left_accum;
  gml_aabb right_accum;

  gml_vec3 axis_lengths;
  gml_vec3_sub(range->centroid_bounds.max, range->centroid_bounds.min, axis_lengths);

  uint32_t bin_count;
  if (thread_data->params->object_binning_mode == GI_BVH_BINNING_MODE_ADAPTIVE)
  {
    bin_count = i32min((uint32_t) (range->stack_size * 0.05f + 4.0f), thread_data->params->object_bin_count);
  }
  else
  {
    bin_count = thread_data->params->object_bin_count;
  }

  struct gi_bvh_object_bin* bins = (struct gi_bvh_object_bin*) thread_data->reused_bins;
  gml_aabb* reused_aabbs = (gml_aabb*) thread_data->reused_aabbs;

  /* Test each axis. */
  for (uint32_t axis = 0; axis < 3; ++axis)
  {
    float axis_length = axis_lengths[axis];

    if (axis_length <= 0.0f)
    {
      continue;
    }

    float k1 = bin_count / axis_length;

    /* Clear object bins. */
    for (uint32_t i = 0; i < bin_count; ++i)
    {
      struct gi_bvh_object_bin* bin = &bins[i];
      gml_aabb_make_smallest(&bin->aabb);
      bin->face_count = 0;
    }

    /* Project faces to bins. */
    for (uint32_t i = 0; i < range->stack_size; ++i)
    {
      const struct gi_bvh_face_ref* ref = &range->stack[(int32_t)i * range->stack_dir];

      float centroid = (ref->aabb.min[axis] + ref->aabb.max[axis]) * 0.5f;

      int32_t bin_index = (int32_t) (k1 * (centroid - range->centroid_bounds.min[axis]));
      bin_index = i32max(0, i32min(bin_index, (bin_count - 1)));

      struct gi_bvh_object_bin* bin = &bins[bin_index];
      bin->face_count++;

      gml_aabb_merge(&bin->aabb, &ref->aabb, &bin->aabb);
    }

    /* Sweep from right to left. */
    gml_aabb_make_smallest(&right_accum);

    for (uint32_t r = bin_count - 1; r > 0; --r)
    {
      const struct gi_bvh_object_bin* bin = &bins[r];
      gml_aabb_merge(&right_accum, &bin->aabb, &right_accum);
      reused_aabbs[r - 1] = right_accum;
    }

    /* Sweep from left to right. */
    gml_aabb_make_smallest(&left_accum);

    uint32_t left_face_count = 0;

    for (uint32_t l = 1; l < bin_count; ++l)
    {
      const struct gi_bvh_object_bin* bin = &bins[l - 1];
      gml_aabb_merge(&left_accum, &bin->aabb, &left_accum);

      left_face_count += bin->face_count;
      uint32_t right_face_count = range->stack_size - left_face_count;

      /* Calculate SAH cost. */
      float area_l = gml_aabb_half_area(&left_accum);
      float area_r = gml_aabb_half_area(&reused_aabbs[l - 1]);

      float sah_cost =
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, left_face_count) * area_l +
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, right_face_count) * area_r;

      /* Abort if cost is higher than best split. */
      if (sah_cost > best_sah_cost)
      {
        continue;
      }

      /* When SAH is equal, prefer equal face distribution. */
      float tie_break = sqrtf((float)left_face_count) + sqrtf((float)right_face_count);

      if (sah_cost == best_sah_cost && tie_break > best_tie_break)
      {
        continue;
      }

      /* Set new best split candidate. */
      gml_aabb overlap_aabb;
      gml_aabb_intersect(&left_accum, &reused_aabbs[l - 1], &overlap_aabb);

      split->sah_cost = sah_cost;
      split->axis = axis;
      split->bin_index = l;
      split->overlap_half_area = gml_aabb_half_area(&overlap_aabb);

      best_sah_cost = sah_cost;
      best_tie_break = tie_break;
    }
  }
}

static void gi_bvh_find_split_spatial(const struct gi_bvh_thread_data* thread_data,
                                      const struct gi_bvh_work_range* range,
                                      struct gi_bvh_split_spatial* split)
{
  struct gi_bvh_spatial_bin* bins = (struct gi_bvh_spatial_bin*) thread_data->reused_bins;
  const struct gi_vertex* vertices = thread_data->params->vertices;
  const struct gi_face* faces = thread_data->params->faces;
  const gml_aabb* range_aabb = &range->aabb_bounds;
  uint32_t bin_count = thread_data->params->spatial_bin_count;

  gml_vec3 axis_lengths;
  gml_aabb_size(&range->aabb_bounds, axis_lengths);
  gml_vec3 bin_sizes;
  gml_vec3_divs(axis_lengths, (float) bin_count, bin_sizes);

  /* Clear spatial bins. */
  for (uint32_t b = 0; b < bin_count * 3; ++b)
  {
    struct gi_bvh_spatial_bin* bin = &bins[b];
    bin->entry_count = 0;
    bin->exit_count = 0;
    gml_aabb_make_smallest(&bin->aabb);
  }

  /* Fill spatial bins. */
  for (uint32_t f = 0; f < range->stack_size; ++f)
  {
    const struct gi_bvh_face_ref* ref = &range->stack[(int32_t) f * range->stack_dir];

    for (uint32_t axis = 0; axis < 3; ++axis)
    {
      if (axis_lengths[axis] <= 0.0f)
      {
        continue;
      }

      float bin_size = bin_sizes[axis];

      const gml_aabb* ref_aabb = &ref->aabb;
      const struct gi_face* face = &faces[ref->index];

      gml_vec3 v_0;
      gml_vec3_assign(vertices[face->v_i[2]].pos, v_0);

      /* Insert all three edges into bin AABBs. */
      for (uint32_t e = 0; e < 3; ++e)
      {
        gml_vec3 v_1, v_start, v_end;
        gml_vec3_assign(vertices[face->v_i[e]].pos, v_1);
        gml_vec3_assign(v_0[axis] <= v_1[axis] ? v_0 : v_1, v_start);
        gml_vec3_assign(v_0[axis] <= v_1[axis] ? v_1 : v_0, v_end);
        gml_vec3_assign(v_1, v_0);

        if (v_start[axis] > range->aabb_bounds.max[axis] ||
            v_end[axis] < range->aabb_bounds.min[axis])
        {
          continue;
        }

        if (v_start[axis] < ref_aabb->min[axis])
        {
          float edge_length = v_end[axis] - v_start[axis];
          float t_plane_rel = (ref_aabb->min[axis] - v_start[axis]) / edge_length;
          gml_vec3_lerp(v_start, v_end, t_plane_rel, v_start);
          v_start[axis] = ref_aabb->min[axis];
        }
        if (v_end[axis] > ref_aabb->max[axis])
        {
          float edge_length = v_end[axis] - v_start[axis];
          float t_plane_rel = (ref_aabb->max[axis] - v_start[axis]) / edge_length;
          gml_vec3_lerp(v_start, v_end, t_plane_rel, v_end);
          v_end[axis] = ref_aabb->max[axis];
        }

        int32_t start_bin_index = (int32_t) ((v_start[axis] - range_aabb->min[axis]) / bin_size);
        int32_t end_bin_index = (int32_t) ((v_end[axis] - range_aabb->min[axis]) / bin_size);
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

        /* Include bin plane intersection points in both bin AABBs. */
        for (int32_t bin_index = start_bin_index; bin_index < end_bin_index; ++bin_index)
        {
          float t_bin_end_plane = range_aabb->min[axis] + (float) (bin_index + 1) * bin_size;

          gml_vec3 v_i;
          float edge_length = v_end[axis] - v_start[axis];
          float t_plane_rel = (t_bin_end_plane - v_start[axis]) / edge_length;
          gml_vec3_lerp(v_start, v_end, t_plane_rel, v_i);
          v_i[axis] = t_bin_end_plane;

          struct gi_bvh_spatial_bin* this_bin = &bins[axis * bin_count + bin_index + 0];
          struct gi_bvh_spatial_bin* next_bin = &bins[axis * bin_count + bin_index + 1];
          gml_aabb_include(&this_bin->aabb, v_i, &this_bin->aabb);
          gml_aabb_include(&next_bin->aabb, v_i, &next_bin->aabb);
        }
      }

      /* Increment entry and exit counters. */
      int32_t start_bin_index = (int32_t) ((ref->aabb.min[axis] - range_aabb->min[axis]) / bin_size);
      int32_t end_bin_index = (int32_t) ((ref->aabb.max[axis] - range_aabb->min[axis]) / bin_size);
      start_bin_index = i32max(0, i32min(start_bin_index, bin_count - 1));
      end_bin_index = i32max(0, i32min(end_bin_index, bin_count - 1));

      bins[axis * bin_count + start_bin_index].entry_count++;
      bins[axis * bin_count + end_bin_index].exit_count++;
    }
  }

  /* Evaluate split planes. */
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

    /* Sweep from right to left. */
    gml_aabb_make_smallest(&right_accum);

    for (int32_t r = bin_count - 1; r > 0; --r)
    {
      const struct gi_bvh_spatial_bin* bin = &bins[axis * bin_count + r];
      gml_aabb_merge(&right_accum, &bin->aabb, &right_accum);
      thread_data->reused_aabbs[r - 1] = right_accum;
    }

    /* Sweep from left to right. */
    gml_aabb_make_smallest(&left_accum);

    uint32_t left_face_count = 0;
    uint32_t right_face_count = range->stack_size;

    for (uint32_t l = 1; l < bin_count; ++l)
    {
      const struct gi_bvh_spatial_bin* bin = &bins[axis * bin_count + l - 1];
      gml_aabb_merge(&left_accum, &bin->aabb, &left_accum);

      left_face_count += bin->entry_count;
      right_face_count -= bin->exit_count;

      /* Ignore invalid splits. */
      if (left_face_count == 0 || right_face_count == 0)
      {
        continue;
      }

      /* Calculate SAH cost. */
      float area_l = gml_aabb_half_area(&left_accum);
      float area_r = gml_aabb_half_area(&thread_data->reused_aabbs[l - 1]);

      float sah_cost =
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, left_face_count) * area_l +
        gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, right_face_count) * area_r;

      /* Abort if cost is higher than best split. */
      if (sah_cost > best_sah_cost)
      {
        continue;
      }

      /* When SAH is equal, prefer equal face distribution. */
      float tie_break = sqrtf((float)left_face_count) + sqrtf((float)right_face_count);

      if (sah_cost == best_sah_cost && tie_break > best_tie_break)
      {
        continue;
      }

      /* Set new best split candidate. */
      split->sah_cost = sah_cost;
      split->axis = axis;
      split->bin_index = l;
      split->left_face_count = left_face_count;
      split->right_face_count = right_face_count;

      best_sah_cost = sah_cost;
      best_tie_break = tie_break;
    }
  }
}

static void gi_bvh_do_split_spatial(const struct gi_bvh_thread_data* thread_data,
                                    const struct gi_bvh_split_spatial* split,
                                    const struct gi_bvh_work_range* range,
                                    struct gi_bvh_work_range* range_left,
                                    struct gi_bvh_work_range* range_right)
{
  uint32_t bin_count = thread_data->params->spatial_bin_count;
  float axis_length = range->aabb_bounds.max[split->axis] - range->aabb_bounds.min[split->axis];
  float bin_size = axis_length / bin_count;

  const struct gi_vertex* vertices = thread_data->params->vertices;
  const struct gi_face* faces = thread_data->params->faces;
  uint32_t axis = split->axis;

  int32_t split_face_count = split->left_face_count + split->right_face_count;
  int32_t free_face_count = range->stack_size_limit - split_face_count;
  assert(free_face_count >= 0);

  struct gi_bvh_work_range* range1 = range->stack_dir == 1 ? range_left : range_right;
  struct gi_bvh_work_range* range2 = range->stack_dir == 1 ? range_right : range_left;

  int32_t range1_index_start = 0;
  int32_t range1_index_end = range->stack_size - 1;
  int32_t range2_index_start = range->stack_size_limit - 1;

  range1->stack_size = 0;
  range1->stack_dir = range->stack_dir;
  range1->stack = range->stack;
  gml_aabb_make_smallest(&range1->aabb_bounds);
  gml_aabb_make_smallest(&range1->centroid_bounds);

  range2->stack_size = 0;
  range2->stack_dir = range->stack_dir * -1;
  range2->stack = range->stack + range2_index_start * range->stack_dir;
  gml_aabb_make_smallest(&range2->aabb_bounds);
  gml_aabb_make_smallest(&range2->centroid_bounds);

  while (range1_index_start <= range1_index_end)
  {
    /* Note that this reference is not const! It will be changed. */
    struct gi_bvh_face_ref* ref = &range->stack[range1_index_start * range->stack_dir];

    const gml_aabb* ref_aabb = &ref->aabb;
    const struct gi_face* face = &faces[ref->index];

    /* Split all edges on the split plane and get AABBs for both sides. */

    gml_aabb left_aabb;
    gml_aabb right_aabb;
    gml_aabb_make_smallest(&left_aabb);
    gml_aabb_make_smallest(&right_aabb);

    gml_vec3 v_0;
    gml_vec3_assign(vertices[face->v_i[2]].pos, v_0);

    for (uint32_t e = 0; e < 3; ++e)
    {
      gml_vec3 v_1, v_start, v_end;
      gml_vec3_assign(vertices[face->v_i[e]].pos, v_1);
      gml_vec3_assign(v_0[split->axis] <= v_1[split->axis] ? v_0 : v_1, v_start);
      gml_vec3_assign(v_0[split->axis] <= v_1[split->axis] ? v_1 : v_0, v_end);
      gml_vec3_assign(v_1, v_0);

      /* Cull and chop edge. */

      if (v_start[split->axis] > range->aabb_bounds.max[split->axis] ||
          v_end[split->axis] < range->aabb_bounds.min[split->axis])
      {
        continue;
      }

      if (v_start[axis] < ref_aabb->min[axis])
      {
        float edge_length = v_end[axis] - v_start[axis];
        float t_plane_rel = (ref_aabb->min[axis] - v_start[axis]) / edge_length;
        gml_vec3_lerp(v_start, v_end, t_plane_rel, v_start);
        v_start[axis] = ref_aabb->min[axis];
      }
      if (v_end[axis] > ref_aabb->max[axis])
      {
        float edge_length = v_end[axis] - v_start[axis];
        float t_plane_rel = (ref_aabb->max[axis] - v_start[axis]) / edge_length;
        gml_vec3_lerp(v_start, v_end, t_plane_rel, v_end);
        v_end[axis] = ref_aabb->max[axis];
      }

      /* Fill left and right AABBs. */

      float t_plane = range->aabb_bounds.min[axis] + split->bin_index * bin_size;

      if (v_start[axis] <= t_plane) { gml_aabb_include(&left_aabb, v_start, &left_aabb); }
      if (v_start[axis] >= t_plane) { gml_aabb_include(&right_aabb, v_start, &right_aabb); }
      if (v_end[axis] <= t_plane) { gml_aabb_include(&left_aabb, v_end, &left_aabb); }
      if (v_end[axis] >= t_plane) { gml_aabb_include(&right_aabb, v_end, &right_aabb); }

      /* Continue if there is no plane intersection. */
      if (t_plane < v_start[axis] || t_plane > v_end[axis] ||
          (t_plane == v_start[axis] && t_plane == v_end[axis]))
      {
        continue;
      }

      /* Otherwise, split into two halves. */

      float edge_length = v_end[axis] - v_start[axis];
      float t_plane_abs = t_plane - v_start[axis];
      float t_plane_rel = (t_plane_abs / edge_length);

      gml_vec3 v_i;
      gml_vec3_lerp(v_start, v_end, t_plane_rel, v_i);
      v_i[axis] = t_plane;

      gml_aabb_include(&left_aabb, v_i, &left_aabb);
      gml_aabb_include(&right_aabb, v_i, &right_aabb);
    }

    gml_aabb_intersect(&left_aabb, &ref->aabb, &left_aabb);
    gml_aabb_intersect(&right_aabb, &ref->aabb, &right_aabb);

    /* Now that we have both side AABBs, we do the actual partitioning. */

    int32_t start_bin_index = (int32_t) ((ref->aabb.min[axis] - range->aabb_bounds.min[axis]) / bin_size);
    int32_t end_bin_index = (int32_t) ((ref->aabb.max[axis] - range->aabb_bounds.min[axis]) / bin_size);
    start_bin_index = i32max(0, i32min(start_bin_index, bin_count - 1));
    end_bin_index = i32max(0, i32min(end_bin_index, bin_count - 1));

    bool is_in_left = start_bin_index < split->bin_index;
    bool is_in_right = end_bin_index >= split->bin_index;

    bool is_in_range1 = (range->stack_dir == 1 && is_in_left) || (range->stack_dir == -1 && is_in_right);
    bool is_in_range2 = (range->stack_dir == 1 && is_in_right) || (range->stack_dir == -1 && is_in_left);
    assert(is_in_range1 || is_in_range2);

    const gml_aabb* new_ref_aabb_r1 = (range->stack_dir == 1) ? &left_aabb : &right_aabb;
    const gml_aabb* new_ref_aabb_r2 = (range->stack_dir == 1) ? &right_aabb : &left_aabb;

    /* Handle face being in the far range. */

    if (is_in_range2)
    {
      range2->stack_size++;

      /* Check if there is space left or if we need to swap. */
      if (range2_index_start != range1_index_end)
      {
        range->stack[range2_index_start * range->stack_dir].index = ref->index;
        range->stack[range2_index_start * range->stack_dir].aabb = *new_ref_aabb_r2;

        /* If the face is not duplicated, pull the next one. */
        if (!is_in_range1)
        {
          range->stack[range1_index_start * range->stack_dir] = range->stack[range1_index_end * range->stack_dir];
          range1_index_end--;
        }
      }
      else
      {
        /* Swap faces and overwrite our AABB with the new, chopped one. */
        struct gi_bvh_face_ref tmp = range->stack[range2_index_start * range->stack_dir];
        range->stack[range2_index_start * range->stack_dir].index = ref->index;
        range->stack[range2_index_start * range->stack_dir].aabb = *new_ref_aabb_r2;
        range->stack[range1_index_start * range->stack_dir] = tmp;
        range1_index_end--;
      }

      const struct gi_bvh_face_ref* new_ref = &range->stack[range2_index_start * range->stack_dir];

      gml_aabb_merge(&range2->aabb_bounds, &new_ref->aabb, &range2->aabb_bounds);

      gml_vec3 new_centroid;
      gml_vec3_add(new_ref->aabb.min, new_ref->aabb.max, new_centroid);
      gml_vec3_muls(new_centroid, 0.5f, new_centroid);
      gml_aabb_include(&range2->centroid_bounds, new_centroid, &range2->centroid_bounds);

      range2_index_start--;
    }

    /* Handle face being in the close range. */

    if (is_in_range1)
    {
      range1->stack_size++;

      ref->aabb = *new_ref_aabb_r1;

      gml_aabb_merge(&range1->aabb_bounds, &ref->aabb, &range1->aabb_bounds);

      gml_vec3 new_centroid;
      gml_vec3_add(ref->aabb.min, ref->aabb.max, new_centroid);
      gml_vec3_muls(new_centroid, 0.5f, new_centroid);
      gml_aabb_include(&range1->centroid_bounds, new_centroid, &range1->centroid_bounds);

      range1_index_start++;
    }
  }

  range1->stack_size_limit = range1->stack_size + free_face_count / 2;
  range2->stack_size_limit = range2->stack_size + free_face_count - free_face_count / 2;

  assert(range_left->stack_size == split->left_face_count);
  assert(range_right->stack_size == split->right_face_count);
}

static void gi_bvh_do_split_object(const struct gi_bvh_split_object* split,
                                   const struct gi_bvh_work_range* range,
                                   struct gi_bvh_work_range* range_left,
                                   struct gi_bvh_work_range* range_right)
{
  /* This partitioning algorithm is a little bit special since we
   * deal with directional ranges. While partitioning, we want both
   * sides to grow inwards from the range bounds. This is performed
   * in-place by taking into account the availability of free space.
   * Left and right face counts are commonly used for partitioning,
   * but because we don't sort in the first place, we can only rely
   * on the centroid position. In case of ambiguities, we compare
   * the referenced face indices. These are unique within a range. */

  /* Range 1 is the side close to the origin of the parent stack.
   * Range 2 represents the opposite side. */

  struct gi_bvh_work_range* range1 = range->stack_dir == 1 ? range_left : range_right;
  struct gi_bvh_work_range* range2 = range->stack_dir == 1 ? range_right : range_left;

  int32_t range1_index_start = 0;
  int32_t range1_index_end = range->stack_size - 1;
  int32_t range2_index_start = range->stack_size_limit - 1;

  /* Reset child ranges. */

  range1->stack_dir = range->stack_dir;
  range1->stack = range->stack;
  range1->stack_size = 0;
  gml_aabb_make_smallest(&range1->aabb_bounds);
  gml_aabb_make_smallest(&range1->centroid_bounds);

  range2->stack_dir = range->stack_dir * -1;
  range2->stack = range->stack + range->stack_dir * range2_index_start;
  range2->stack_size = 0;
  gml_aabb_make_smallest(&range2->aabb_bounds);
  gml_aabb_make_smallest(&range2->centroid_bounds);

  /* Do partitioning. */

  bool stack_dir_pos = range->stack_dir == +1;
  bool stack_dir_neg = range->stack_dir == -1;

  while (range1_index_start <= range1_index_end)
  {
    const struct gi_bvh_face_ref* ref = &range->stack[range1_index_start * range->stack_dir];

    gml_vec3 centroid;
    gml_vec3_add(ref->aabb.min, ref->aabb.max, centroid);

    bool is_in_left = (centroid[split->axis] < split->dcentroid) ||
                      (centroid[split->axis] == split->dcentroid && ref->index <= split->face_index);

    bool is_in_range1 = (stack_dir_pos && is_in_left) || (stack_dir_neg && !is_in_left);

    gml_vec3_muls(centroid, 0.5f, centroid);

    /* Handle face being in the close range. */

    if (is_in_range1)
    {
      range1->stack_size++;

      gml_aabb_include(&range1->centroid_bounds, centroid, &range1->centroid_bounds);
      gml_aabb_merge(&range1->aabb_bounds, &ref->aabb, &range1->aabb_bounds);

      range1_index_start++;
      continue;
    }

    /* Handle face being in the far range. */

    range2->stack_size++;

    gml_aabb_include(&range2->centroid_bounds, centroid, &range2->centroid_bounds);
    gml_aabb_merge(&range2->aabb_bounds, &ref->aabb, &range2->aabb_bounds);

    /* Check if there is space left or if we need to swap. */
    if (range2_index_start != range1_index_end)
    {
      range->stack[range2_index_start * range->stack_dir] = *ref;
      range->stack[range1_index_start * range->stack_dir] = range->stack[range1_index_end * range->stack_dir];
    }
    else
    {
      struct gi_bvh_face_ref tmp = range->stack[range2_index_start * range->stack_dir];
      range->stack[range2_index_start * range->stack_dir] = range->stack[range1_index_start * range->stack_dir];
      range->stack[range1_index_start * range->stack_dir] = tmp;
    }

    range2_index_start--;
    range1_index_end--;
  }

  assert(range1->stack_size > 0);
  assert(range2->stack_size > 0);

  /* Assign stack limits. */

  int32_t free_face_count = range->stack_size_limit - range->stack_size;
  int32_t half_free_face_count = free_face_count / 2;
  range1->stack_size_limit = range1->stack_size + half_free_face_count;
  range2->stack_size_limit = range2->stack_size + (free_face_count - half_free_face_count);
}

static void gi_bvh_do_split_object_binned(const struct gi_bvh_thread_data* thread_data,
                                          const struct gi_bvh_split_object_binned* split,
                                          const struct gi_bvh_work_range* range,
                                          struct gi_bvh_work_range* range_left,
                                          struct gi_bvh_work_range* range_right)
{
  /* See non-binned object splitting for a general algorithm description. */

  struct gi_bvh_work_range* range1 = range->stack_dir == 1 ? range_left : range_right;
  struct gi_bvh_work_range* range2 = range->stack_dir == 1 ? range_right : range_left;

  int32_t range1_index_start = 0;
  int32_t range1_index_end = range->stack_size - 1;
  int32_t range2_index_start = range->stack_size_limit - 1;

  /* Reset child ranges. */

  range1->stack_dir = range->stack_dir;
  range1->stack = range->stack;
  range1->stack_size = 0;
  gml_aabb_make_smallest(&range1->aabb_bounds);
  gml_aabb_make_smallest(&range1->centroid_bounds);

  range2->stack_dir = range->stack_dir * -1;
  range2->stack = range->stack + range->stack_dir * range2_index_start;
  range2->stack_size = 0;
  gml_aabb_make_smallest(&range2->aabb_bounds);
  gml_aabb_make_smallest(&range2->centroid_bounds);

  /* Precalculate values. */

  float axis_length = range->centroid_bounds.max[split->axis] - range->centroid_bounds.min[split->axis];

  uint32_t bin_count;
  if (thread_data->params->object_binning_mode == GI_BVH_BINNING_MODE_ADAPTIVE)
  {
    bin_count = i32min((uint32_t) (range->stack_size * 0.05f + 4.0f), thread_data->params->object_bin_count);
  }
  else
  {
    bin_count = thread_data->params->object_bin_count;
  }

  float k1 = bin_count / axis_length;
  bool stack_dir_pos = range->stack_dir == +1;
  bool stack_dir_neg = range->stack_dir == -1;

  /* Do partitioning. */

  while (range1_index_start <= range1_index_end)
  {
    /* Determining which side a face is on is done by re-projecting it
     * to its bin and comparing the bin index to the split bin index.
     * This should be consistent with the way we found the split - this
     * way, we don't have any deviations due to floating point math. */

    const struct gi_bvh_face_ref* ref = &range->stack[range1_index_start * range->stack_dir];

    gml_vec3 centroid;
    gml_vec3_add(ref->aabb.min, ref->aabb.max, centroid);
    gml_vec3_muls(centroid, 0.5f, centroid);

    int32_t bin_index = (int32_t) (k1 * (centroid[split->axis] - range->centroid_bounds.min[split->axis]));
    bin_index = i32max(0, i32min(bin_index, (bin_count - 1)));

    bool is_in_range1 = (stack_dir_pos && bin_index < split->bin_index) ||
                        (stack_dir_neg && bin_index >= split->bin_index);

    /* Handle face being in the close range. This partitioning algorithm
     * is essentially the same as in the non-binned object split. */

    if (is_in_range1)
    {
      range1->stack_size++;

      gml_aabb_include(&range1->centroid_bounds, centroid, &range1->centroid_bounds);
      gml_aabb_merge(&range1->aabb_bounds, &ref->aabb, &range1->aabb_bounds);

      range1_index_start++;
      continue;
    }

    /* Handle face being in the far range. */

    range2->stack_size++;

    gml_aabb_include(&range2->centroid_bounds, centroid, &range2->centroid_bounds);
    gml_aabb_merge(&range2->aabb_bounds, &ref->aabb, &range2->aabb_bounds);

    /* Check if there is space left or if we need to swap. */
    if (range2_index_start != range1_index_end)
    {
      range->stack[range2_index_start * range->stack_dir] = *ref;
      range->stack[range1_index_start * range->stack_dir] = range->stack[range1_index_end * range->stack_dir];
    }
    else
    {
      struct gi_bvh_face_ref tmp = range->stack[range2_index_start * range->stack_dir];
      range->stack[range2_index_start * range->stack_dir] = range->stack[range1_index_start * range->stack_dir];
      range->stack[range1_index_start * range->stack_dir] = tmp;
    }

    range2_index_start--;
    range1_index_end--;
  }

  assert(range1->stack_size > 0);
  assert(range2->stack_size > 0);

  /* Assign stack limits. */

  int32_t free_face_count = range->stack_size_limit - range->stack_size;
  int32_t half_free_face_count = free_face_count / 2;
  range1->stack_size_limit = range1->stack_size + half_free_face_count;
  range2->stack_size_limit = range2->stack_size + (free_face_count - half_free_face_count);
}

static bool gi_bvh_build_work_range(const struct gi_bvh_thread_data* thread_data,
                                    const struct gi_bvh_work_range* range,
                                    struct gi_bvh_work_range* range_left,
                                    struct gi_bvh_work_range* range_right)
{
  /* Make a leaf if face count is too low. */

  if (range->stack_size == 1)
  {
    return false;
  }

  /* Check if we want to use binning. */

  bool is_cb_degenerate;
  {
    gml_vec3 cbmin, cbmax;
    gml_vec3_assign(range->centroid_bounds.min, cbmin);
    gml_vec3_assign(range->centroid_bounds.max, cbmax);

    is_cb_degenerate = (cbmin[0] == cbmax[0] && cbmin[1] == cbmax[1]) ||
                       (cbmin[1] == cbmax[1] && cbmin[2] == cbmax[2]) ||
                       (cbmin[2] == cbmax[2] && cbmin[0] == cbmax[0]);
  }

  bool should_use_binning = range->stack_size > thread_data->params->object_binning_threshold;
  bool is_binning_enabled = thread_data->params->object_binning_mode != GI_BVH_BINNING_MODE_OFF;

  bool do_binning = is_binning_enabled && !is_cb_degenerate && should_use_binning;

  /* Evaluate possible splits. */

  struct gi_bvh_split_object_binned split_object_binned;
  struct gi_bvh_split_object split_object;
  struct gi_bvh_split_spatial split_spatial;

  split_object_binned.sah_cost = INFINITY;
  split_object_binned.bin_index = 0;
  split_object_binned.overlap_half_area = 0.0f;
  split_object.sah_cost = INFINITY;
  split_object.overlap_half_area = 0.0f;
  split_spatial.sah_cost = INFINITY;

  float overlap_half_area;

  if (do_binning)
  {
    gi_bvh_find_split_object_binned(thread_data, range, &split_object_binned);
    overlap_half_area = split_object_binned.overlap_half_area;
  }
  else
  {
    gi_bvh_find_split_object(thread_data, range, &split_object);
    overlap_half_area = split_object.overlap_half_area;
  }

  bool try_spatial_split = (overlap_half_area / thread_data->root_half_area) > thread_data->params->spatial_split_alpha;

  if (try_spatial_split)
  {
    gi_bvh_find_split_spatial(thread_data, range, &split_spatial);
  }

  float face_test_cost = gi_bvh_face_test_cost(thread_data->params->face_intersection_cost, thread_data->params->face_batch_size, range->stack_size);
  float leaf_sah_cost = face_test_cost * gml_aabb_half_area(&range->aabb_bounds);

  /* Find best split option. */

  float best_sah_cost = fminf(
    fminf(split_object.sah_cost, split_object_binned.sah_cost),
    fminf(split_spatial.sah_cost, leaf_sah_cost)
  );

  /* Handle best split option. */

  bool fits_in_leaf = range->stack_size <= thread_data->params->leaf_max_face_count;

  if (fits_in_leaf && best_sah_cost == leaf_sah_cost)
  {
    return false;
  }

  if (best_sah_cost == split_spatial.sah_cost)
  {
    int32_t split_face_count = split_spatial.left_face_count + split_spatial.right_face_count;
    int32_t free_face_count = ((int32_t) range->stack_size_limit) - split_face_count;

    if (free_face_count >= 0)
    {
      gi_bvh_do_split_spatial(thread_data, &split_spatial, range, range_left, range_right);
      return true;
    }

    /* Fall back to reserve buffer if no space is left in range. This is usually only the case
     * for ranges with few faces. Therefore it does not make sense to allocate more than needed. */
    free_face_count = *thread_data->reserve_buffer_capacity;

    if (free_face_count >= split_face_count)
    {
      *thread_data->reserve_buffer_capacity -= split_face_count;

      struct gi_bvh_work_range reserve_range;
      reserve_range.stack = &thread_data->reserve_buffer[free_face_count - split_face_count];
      reserve_range.stack_dir = 1;
      reserve_range.stack_size = range->stack_size;
      reserve_range.aabb_bounds = range->aabb_bounds;
      reserve_range.centroid_bounds = range->centroid_bounds;
      reserve_range.stack_size_limit = split_face_count;

      memcpy(
        reserve_range.stack,
        (range->stack_dir == 1) ? range->stack : (range->stack - (range->stack_size - 1)),
        range->stack_size * sizeof(struct gi_bvh_face_ref)
      );

      gi_bvh_do_split_spatial(thread_data, &split_spatial, &reserve_range, range_left, range_right);
      return true;
    }
  }

  if (do_binning)
  {
    gi_bvh_do_split_object_binned(thread_data, &split_object_binned, range, range_left, range_right);
  }
  else
  {
    gi_bvh_do_split_object(&split_object, range, range_left, range_right);
  }

  return true;
}

void gi_bvh_build(const struct gi_bvh_params* params,
                  struct gi_bvh* bvh)
{
  /* Set up root work range. */

  gml_aabb root_aabb_bounds;
  gml_aabb root_centroid_bounds;
  gml_aabb_make_smallest(&root_aabb_bounds);
  gml_aabb_make_smallest(&root_centroid_bounds);

  uint32_t root_stack_size_limit = params->face_count;
  if (params->spatial_split_alpha < 1.0f)
  {
    root_stack_size_limit *= 2;
  }

  struct gi_bvh_face_ref* root_stack = (struct gi_bvh_face_ref*) malloc(root_stack_size_limit * sizeof(struct gi_bvh_face_ref));

  uint32_t root_stack_size = 0;

  for (uint32_t i = 0; i < params->face_count; ++i)
  {
    const struct gi_face* face = &params->faces[i];
    const struct gi_vertex* v_a = &params->vertices[face->v_i[0]];
    const struct gi_vertex* v_b = &params->vertices[face->v_i[1]];
    const struct gi_vertex* v_c = &params->vertices[face->v_i[2]];

    struct gi_bvh_face_ref* face_ref = &root_stack[root_stack_size];

    gml_aabb_make_from_triangle(v_a->pos, v_b->pos, v_c->pos, &face_ref->aabb);

    bool is_tri_degenerate =
      (face_ref->aabb.min[0] == face_ref->aabb.max[0] && face_ref->aabb.min[1] == face_ref->aabb.max[1]) ||
      (face_ref->aabb.min[1] == face_ref->aabb.max[1] && face_ref->aabb.min[2] == face_ref->aabb.max[2]) ||
      (face_ref->aabb.min[2] == face_ref->aabb.max[2] && face_ref->aabb.min[0] == face_ref->aabb.max[0]);

    if (is_tri_degenerate)
    {
      continue;
    }

    face_ref->index = i;

    gml_aabb_merge(&root_aabb_bounds, &face_ref->aabb, &root_aabb_bounds);

    gml_vec3 centroid;
    gml_vec3_add(face_ref->aabb.max, face_ref->aabb.min, centroid);
    gml_vec3_muls(centroid, 0.5f, centroid);

    gml_aabb_include(&root_centroid_bounds, centroid, &root_centroid_bounds);

    root_stack_size++;
  }

  float root_half_area = gml_aabb_half_area(&root_aabb_bounds);

  uint32_t max_job_stack_size = params->face_count;

  struct gi_bvh_work_job* job_stack = (struct gi_bvh_work_job*) malloc(max_job_stack_size * sizeof(struct gi_bvh_work_job));

  job_stack[0].range.stack            = root_stack;
  job_stack[0].range.stack_dir        = 1;
  job_stack[0].range.stack_size       = root_stack_size;
  job_stack[0].range.stack_size_limit = root_stack_size_limit;
  job_stack[0].range.aabb_bounds      = root_aabb_bounds;
  job_stack[0].range.centroid_bounds  = root_centroid_bounds;
  job_stack[0].node_index             = 0;

  int32_t job_stack_size = 1;

  /* Set up bvh. */

  uint32_t max_face_count = (params->spatial_split_alpha < 1.0f) ? params->face_count * 8 : params->face_count;
  uint32_t max_node_count = max_face_count * 2;

  bvh->aabb = root_aabb_bounds;
  bvh->face_count = 0;
  bvh->faces = malloc(max_face_count * sizeof(struct gi_face));
  bvh->node_count = 1;
  bvh->nodes = malloc(max_node_count * sizeof(struct gi_bvh_node));

  /* Allocate thread-local memory. */

  uint32_t object_bins_size = params->object_bin_count * sizeof(struct gi_bvh_object_bin);
  uint32_t spatial_bins_size = (params->spatial_split_alpha == 1.0f) ? 0 : (params->spatial_bin_count * sizeof(struct gi_bvh_spatial_bin) * 3);
  uint32_t reused_bins_size = imax(object_bins_size, spatial_bins_size);

  int32_t reserve_buffer_capacity = (int32_t) (params->spatial_reserve_factor * params->face_count);
  struct gi_bvh_face_ref* reserve_buffer = (struct gi_bvh_face_ref*) malloc(reserve_buffer_capacity * sizeof(struct gi_bvh_face_ref));

  int32_t reused_aabbs_size = imax(params->spatial_bin_count, params->face_count) * sizeof(gml_aabb);

  struct gi_bvh_thread_data thread_data;
  thread_data.params = params;
  thread_data.reused_bins = (void*) malloc(reused_bins_size);
  thread_data.reused_aabbs = (gml_aabb*) malloc(reused_aabbs_size);
  thread_data.root_half_area = root_half_area;
  thread_data.reserve_buffer = reserve_buffer;
  thread_data.reserve_buffer_capacity = &reserve_buffer_capacity;

  /* Perform work until queue is empty. */

  while (job_stack_size > 0)
  {
    job_stack_size--;

    /* Process bvh range. */
    struct gi_bvh_work_job job = job_stack[job_stack_size];

    struct gi_bvh_work_range left_range;
    struct gi_bvh_work_range right_range;

    bool make_leaf = !gi_bvh_build_work_range(&thread_data, &job.range, &left_range, &right_range);

    struct gi_bvh_node* node = &bvh->nodes[job.node_index];

    node->aabb = job.range.aabb_bounds;

    /* We did not split the range, make a leaf instead. */
    if (make_leaf)
    {
      node->field1 = bvh->face_count;
      node->field2 = 0x80000000 | job.range.stack_size;

      /* Resolve face references and write them into the bvh. */
      for (uint32_t i = 0; i < job.range.stack_size; ++i)
      {
        const struct gi_bvh_face_ref* ref = &job.range.stack[(int32_t)i * job.range.stack_dir];
        const struct gi_face* face = &params->faces[ref->index];
        bvh->faces[bvh->face_count] = *face;
        bvh->face_count++;
      }

      continue;
    }

    /* Otherwise, create two new nodes. */
    node->field1 = bvh->node_count;
    bvh->node_count++;
    node->field2 = bvh->node_count;
    bvh->node_count++;

    /* Enqueue new subranges. */
    job_stack[job_stack_size].node_index = node->field1;
    job_stack[job_stack_size].range = left_range;
    job_stack_size++;

    job_stack[job_stack_size].node_index = node->field2;
    job_stack[job_stack_size].range = right_range;
    job_stack_size++;
  }

  /* Free memory. */

  free(thread_data.reused_bins);
  free(thread_data.reused_aabbs);
  free(reserve_buffer);

  free(job_stack);
  free(root_stack);

  /* Reallocate bvh memory. */

  bvh->nodes = (struct gi_bvh_node*) realloc(bvh->nodes, bvh->node_count * sizeof(struct gi_bvh_node));
  bvh->faces = (struct gi_face*) realloc(bvh->faces, bvh->face_count * sizeof(struct gi_face));
}

void gi_free_bvh(struct gi_bvh* bvh)
{
  free(bvh->nodes);
  free(bvh->faces);
}
